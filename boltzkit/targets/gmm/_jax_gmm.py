import jax
import jax.numpy as jnp


class JaxGMM:
    def __init__(self, means: jnp.ndarray, scales: jnp.ndarray, logits: jnp.ndarray):
        """
        means: (K, D)
        scales: (K, D)
        logits: (K,)
        """
        self.means = means
        self.scales = scales
        self.logits = logits

        self.n_components, self.dim = means.shape

        # Precompute for log_prob
        self.log_weights = jax.nn.log_softmax(self.logits)  # (K,)
        self.log_norm_consts = -0.5 * self.dim * jnp.log(2 * jnp.pi) - jnp.sum(
            jnp.log(self.scales), axis=1
        )  # (K,)

    def _component_log_prob(self, x):
        """
        x: (..., D)
        returns: (..., K)
        """
        x_exp = jnp.expand_dims(x, axis=-2)  # (..., 1, D)

        diff = (x_exp - self.means) / self.scales  # (..., K, D)
        quad = -0.5 * jnp.sum(diff**2, axis=-1)  # (..., K)

        return quad + self.log_norm_consts  # (..., K)

    def log_prob(self, x: jnp.ndarray):
        """
        x: (..., D)
        returns: (...,)
        """
        comp_lp = self._component_log_prob(x)

        return jax.scipy.special.logsumexp(comp_lp + self.log_weights, axis=-1)

    def sample(self, key: jax.Array, n_samples: int):
        """
        Sample from the mixture.
        """
        key_cat, key_norm = jax.random.split(key)

        comp_ids = jax.random.categorical(
            key_cat, self.logits, shape=(n_samples,)
        )  # (N,)

        means = self.means[comp_ids]  # (N, D)
        scales = self.scales[comp_ids]  # (N, D)

        eps = jax.random.normal(key_norm, shape=(n_samples, self.dim))

        return means + eps * scales


import jax
import jax.numpy as jnp
import jax.scipy


def make_jax_gmm_log_prob(means: jnp.ndarray, scales: jnp.ndarray, logits: jnp.ndarray):
    """
    Factory to create a JIT-compiled, batched log_prob function for a Gaussian Mixture Model.

    means: (K, D)
    scales: (K, D)
    logits: (K,)

    Returns:
        log_prob_fn: function taking x of shape (N, D) or (D,) -> returns (N,) or scalar
    """
    # Ensure proper shapes
    means = jnp.atleast_2d(means)  # (K, D)
    scales = jnp.atleast_2d(scales)  # (K, D)
    logits = jnp.atleast_1d(logits)  # (K,)
    n_components, dim = means.shape

    if scales.shape != (n_components, dim):
        raise ValueError(f"scales must have shape (K, D), got {scales.shape}")
    if logits.shape[0] != n_components:
        raise ValueError(f"logits must have shape (K,), got {logits.shape}")

    # Precompute constants
    log_weights = jax.nn.log_softmax(logits)  # (K,)
    log_norm_consts = -0.5 * dim * jnp.log(2 * jnp.pi) - jnp.sum(
        jnp.log(scales), axis=1
    )  # (K,)

    # Component log-probability for a single x
    def component_log_prob(x):
        # x: (D,)
        diff = (x - means) / scales  # (K, D)
        quad = -0.5 * jnp.sum(diff**2, axis=-1)  # (K,)
        return quad + log_norm_consts  # (K,)

    # Single-sample log_prob
    def log_prob_single(x):
        comp_lp = component_log_prob(x)  # (K,)
        return jax.scipy.special.logsumexp(comp_lp + log_weights)  # scalar

    return log_prob_single
