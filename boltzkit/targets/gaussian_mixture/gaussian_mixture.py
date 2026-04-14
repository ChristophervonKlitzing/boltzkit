import numpy as np
from scipy.special import logsumexp

from boltzkit.targets.base import DispatchedTarget


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

    # def load_dataset(self, T: float, n_samples: int, **kwargs): ...


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
