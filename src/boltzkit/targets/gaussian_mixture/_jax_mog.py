import jax
import jax.numpy as jnp
from boltzkit.targets.base.dispatched_eval.jax import JaxEval


def create_jax_MoG_eval(means: jnp.ndarray, scales: jnp.ndarray, logits: jnp.ndarray):
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
    def log_prob_single(x: jax.Array):
        comp_lp = component_log_prob(x)  # (K,)
        return jax.scipy.special.logsumexp(comp_lp + log_weights)  # scalar

    return JaxEval.create_from_log_prob_single(log_prob_single)
