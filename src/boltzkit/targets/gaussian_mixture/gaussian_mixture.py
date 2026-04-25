from typing import Literal

import numpy as np
from scipy.special import logsumexp

from boltzkit.targets.base import DispatchedTarget
from boltzkit.utils.dataset import Dataset


class DiagonalGaussianMixture(DispatchedTarget):
    """
    Gaussian Mixture Model (GMM) with diagonal covariance matrices.

    Each component is a multivariate normal distribution with independent
    dimensions, i.e., a diagonal covariance matrix. The covariance of each
    component k is given by:

    .. math::
        Σ_k = diag(σ_{k,1}^2, ..., σ_{k,D}^2)

    where `diag_stds[k, d] = σ_{k,d}` are the standard deviations.

    The mixture weights are parameterized by `logits`, which are internally
    normalized to log-probabilities.
    """

    def __init__(
        self,
        means: np.ndarray,
        diag_stds: np.ndarray,
        logits: np.ndarray,
    ):
        """
            Initialize a diagonal-covariance Gaussian mixture model.

                Parameters
        ----------
        means : np.ndarray, shape (K, D)
            Mean vectors of the mixture components, where K is the number of
            components and D is the dimensionality.
        diag_stds : np.ndarray, shape (K, D)
            Per-component, per-dimension standard deviations (not variances).
            Must be strictly positive.
        logits : np.ndarray, shape (K,)
            Unnormalized log-weights of the mixture components. These are
            normalized internally via log-softmax.

        Raises
        ------
        ValueError
            If input shapes are inconsistent or invalid.
        """

        # Basic dimensionality check first (prevents cryptic index errors)
        if means.ndim != 2:
            raise ValueError(
                f"`means` must be a 2D array of shape (n_components, dim), "
                f"but got array with shape {means.shape}."
            )

        dim = means.shape[1]
        super().__init__(dim)

        self._n_components = means.shape[0]

        self._means = means
        self._scales = diag_stds
        self._logits = logits - logsumexp(logits)  # normalize

        # Shape consistency checks
        if diag_stds.shape != means.shape:
            raise ValueError(
                f"`scales` must have the same shape as `means`. "
                f"Expected shape {means.shape}, but got {diag_stds.shape}."
            )

        if logits.shape != (self._n_components,):
            raise ValueError(
                f"`logits` must be a 1D array with length equal to the number "
                f"of components ({self._n_components}), but got shape {logits.shape}."
            )

    @classmethod
    def create_isotropic(
        cls,
        means: np.ndarray,
        std: float,
        logits: np.ndarray,
    ):
        """
        Construct a GMM with isotropic components.

        Each component shares the same standard deviation across all
        dimensions, i.e., σ_{k,d} = std for all k, d.

        Parameters
        ----------
        means : np.ndarray, shape (K, D)
            Mean vectors of the mixture components.
        std : float
            Shared standard deviation for all components and dimensions.
            Must be positive.
        logits : np.ndarray, shape (K,)
            Unnormalized log-weights of the mixture components.

        Returns
        -------
        DiagonalGaussianMixture
            A GMM instance with diagonal covariance where all diagonal
            entries are equal per component.

        Raises
        ------
        ValueError
            If `std` is not positive or inputs are invalid.
        """
        if std <= 0.0:
            raise ValueError(
                f"Standard deviation must be greater than zero but got {std=}"
            )
        scales = np.ones_like(means) * std
        return DiagonalGaussianMixture(means, scales, logits)

    @classmethod
    def create_isotropic_uniform(
        cls,
        std: float,
        n_components: int,
        dim: int,
        mean_range: tuple[float, float],
        seed: int = 0,
    ):
        """
        Construct an isotropic GMM with uniformly sampled means and equal weights.

        Component means are sampled uniformly from the given range, and all
        mixture weights are set to be equal.

        Parameters
        ----------
        std : float
            Shared standard deviation for all components and dimensions.
            Must be positive.
        n_components : int
            Number of mixture components (K).
        dim : int
            Dimensionality of each component (D).
        mean_range : tuple of float
            Lower and upper bounds (min, max) for uniform sampling of means.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        DiagonalGaussianMixture
            A GMM instance with:
            - uniformly distributed means
            - isotropic covariance
            - uniform mixture weights

        Notes
        -----
        The mixture weights are initialized as uniform, i.e.,
        p(k) = 1 / K for all components.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        rng = np.random.default_rng(seed)
        means = rng.uniform(*mean_range, size=(n_components, dim))
        logits = np.full((n_components,), fill_value=-np.log(n_components))
        return cls.create_isotropic(means=means, std=std, logits=logits)

    @classmethod
    def create_gmm40(
        cls,
        dim: int = 2,
        n_components: int = 40,
        loc_scaling: float = 40.0,
        scale: float = 1.0,
        seed: int = 0,
    ):
        return cls.create_isotropic_uniform(
            std=scale,
            n_components=n_components,
            dim=dim,
            mean_range=(-loc_scaling, loc_scaling),
            seed=seed,
        )

    def can_sample(self):
        return True

    def sample(self, n_samples, seed: int | None = None):
        rng = np.random.default_rng(seed)

        means = self._means
        scales = self._scales
        logits = self._logits

        K, D = means.shape

        probs = np.exp(logits)

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

        component_probs = np.exp(logits)

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

    diag_stds = np.array(
        [
            [0.8, 0.8],
            [0.5, 0.9],
            [2.0, 0.6],
        ]
    )

    logits = np.log(np.array([0.4, 0.4, 0.2]))

    target = DiagonalGaussianMixture(means, diag_stds, logits)
    target = DiagonalGaussianMixture.create_gmm40()

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
    x = np.linspace(-50, 50, 200)
    y = np.linspace(-50, 50, 200)
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

    plt.contourf(X, Y, np.log(probs), levels=50)

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
