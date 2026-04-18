from typing import Literal

import numpy as np
from scipy.special import logsumexp

from boltzkit.targets.base import DispatchedTarget
from boltzkit.utils.dataset import Dataset


class DiagonalGaussianMixture(DispatchedTarget):
    def __init__(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        logits: np.ndarray,
    ):
        dim = means.shape[1]
        super().__init__(dim)

        self._means = means
        self._scales = scales
        self._logits = logits

    def can_sample(self):
        return True

    def sample(self, n_samples, seed: int | None = None):
        rng = np.random.default_rng(seed)

        means = self._means
        scales = self._scales
        logits = self._logits

        K, D = means.shape

        # Normalize mixture weights
        log_probs = logits - logsumexp(logits)
        probs = np.exp(log_probs)

        # Sample all components at once
        components = rng.choice(K, size=n_samples, p=probs)  # (N,)

        # Gather parameters per sample (fully vectorized)
        chosen_means = means[components]  # (N, D)
        chosen_scales = scales[components]  # (N, D)

        # Sample noise
        eps = rng.standard_normal(size=(n_samples, D)).astype(np.float32)

        # Reparameterized sampling
        samples = chosen_means + chosen_scales * eps

        return samples

    def load_dataset(
        self,
        type: Literal["train", "val", "test", "seed"],
        length: int,
        *,
        include_samples: bool = True,
        include_energies: bool = False,
        include_forces: bool = False,
        seed: int = 0,
    ) -> Dataset:
        """
        Generate a deterministic synthetic dataset of a given split and size.

        Each combination of `type` and `seed` defines a unique infinite
        pseudo-random dataset. The returned dataset corresponds to the first
        `length` elements (a prefix) of that deterministic sequence.

        In particular, for any fixed `type` and `seed`:

            load_dataset(type, n, seed=seed)
            == load_dataset(type, n + k, seed=seed)[:n]

        for all n >= 0 and k >= 0.

        This means increasing `length` extends the same underlying dataset
        rather than generating an unrelated new sample.

        Parameters
        ----------
        type : {"train", "val", "test"}
            Dataset split identifier. Different splits produce different
            deterministic dataset streams.

        length : int
            Number of samples to generate (the prefix length).

        seed : int, optional
            Additional seed used to select a reproducible variant of the
            dataset within the chosen split. Default is 0.

        Returns
        -------
        Dataset
            A dataset containing the first `length` elements of the
            deterministic sequence defined by (`type`, `seed`).

        Notes
        -----
        - The dataset is fully reproducible for a fixed (`type`, `seed`) pair.
        - Different `type` values generate independent dataset streams.
        - Increasing `length` preserves all previously generated samples.
        """

        split_ids = {"train": 0, "val": 1, "test": 2}
        seeds = np.random.default_rng(seed).integers(0, 2**32, size=3)
        root_seed = seeds[split_ids[type]]

        ss = np.random.SeedSequence(root_seed)

        # two independent child streams
        ss_comp, ss_eps = ss.spawn(2)

        rng_comp = np.random.default_rng(ss_comp)
        rng_eps = np.random.default_rng(ss_eps)

        means = self._means
        scales = self._scales
        logits = self._logits

        K, D = means.shape

        # normalize mixture weights
        component_log_probs = logits - logsumexp(logits)
        component_probs = np.exp(component_log_probs)

        # component sampling (stream 1)
        components = rng_comp.choice(K, size=length, p=component_probs)

        # parameter lookup (fully vectorized)
        chosen_means = means[components]
        chosen_scales = scales[components]

        # noise sampling (stream 2)
        eps = rng_eps.standard_normal((length, D)).astype(np.float32)

        samples = chosen_means + chosen_scales * eps

        if include_energies:
            log_probs = self.get_log_prob(samples)
        else:
            log_probs = None
        if include_forces:
            scores = self.get_score(samples)
        else:
            scores = None

        if not include_samples:
            samples = None

        return Dataset(kB_T=1.0, samples=samples, log_probs=log_probs, scores=scores)

    def _create_numpy_eval(self):
        from boltzkit.targets.gaussian_mixture._np_mog import NumpyMoG

        return NumpyMoG(self._means, self._scales, self._logits)

    def _create_torch_eval(self):
        from boltzkit.targets.gaussian_mixture._torch_mog import TorchMoG

        return TorchMoG(self._means, self._scales, self._logits)

    def _create_jax_eval(self):
        from boltzkit.targets.gaussian_mixture._jax_mog import (
            create_jax_MoG_eval,
        )
        from boltzkit.targets._jax import make_eval_from_jax_log_prob_single

        return make_eval_from_jax_log_prob_single(
            create_jax_MoG_eval(self._means, self._scales, self._logits)
        )


class IsotropicGaussianMixture(DiagonalGaussianMixture):
    def __init__(self, means: np.ndarray, scale: float, logits: np.ndarray):
        scales = np.ones_like(means) * scale
        super().__init__(means, scales, logits)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # -----------------------
    # Create toy mixture
    # -----------------------
    means = np.array(
        [
            [-2.0, -2.0],
            [2.0, 2.0],
            [2.0, -2.0],
        ]
    )

    scales = np.array(
        [
            [0.8, 0.8],
            [0.5, 0.9],
            [2.0, 0.6],
        ]
    )

    logits = np.log(np.array([0.4, 0.4, 0.2]))

    target = DiagonalGaussianMixture(means, scales, logits)

    d1 = target.load_dataset(type="val", length=5, include_energies=True)
    d2 = target.load_dataset(type="val", length=10)

    print(d1.get_samples())
    print(d2.get_samples())

    print(d1.get_log_probs())
    print(target.get_log_prob(d1.get_samples()))

    # -----------------------
    # Sample points
    # -----------------------
    samples = target.sample(500, seed=0)

    # -----------------------
    # Grid for density
    # -----------------------
    x = np.linspace(-6, 6, 200)
    y = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x, y)

    grid = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # -----------------------
    # Compute density
    # -----------------------
    log_probs = target.get_log_prob(grid)
    probs = np.exp(np.array(log_probs).reshape(X.shape))

    # -----------------------
    # Plot
    # -----------------------
    plt.figure(figsize=(7, 6))

    plt.contourf(X, Y, probs, levels=50)

    plt.scatter(
        samples[:, 0],
        samples[:, 1],
        s=5,
        alpha=0.4,
        marker="x",
        color="red",
    )

    plt.title("Gaussian Mixture: density + samples")
    plt.tight_layout()
    plt.show()
