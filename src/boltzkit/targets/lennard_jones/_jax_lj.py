import jax
import jax.numpy as jnp
from boltzkit.targets.base.dispatched_eval.jax import JaxEval


def create_jax_lennard_jones_eval(
    n_particles: int,
    spatial_dims: int = 3,
    eps: float = 1.0,
    rm: float = 1.0,
    oscillator: bool = True,
    oscillator_scale: float = 1.0,
    energy_factor: float = 1.0,
):
    def energy_single(x: jax.Array):
        # (D,) → (P, S)
        x = x.reshape(n_particles, spatial_dims)

        # pairwise differences (P, P, S)
        diff = x[:, None, :] - x[None, :, :]

        # squared distances (P, P)
        dist_sq = jnp.sum(diff**2, axis=-1) + 1e-6
        dists = jnp.sqrt(dist_sq)

        # Lennard-Jones
        inv_r6 = (rm / dists) ** 6
        lj = eps * (inv_r6**2 - 2.0 * inv_r6)

        # remove diagonal safely
        lj = lj * (1.0 - jnp.eye(n_particles))

        # sum all interactions (double-counted, like PyTorch)
        lj_energy = jnp.sum(lj) * energy_factor

        # oscillator term
        if oscillator:
            centered = x - jnp.mean(x, axis=0, keepdims=True)
            osc_energy = 0.5 * jnp.sum(centered**2)
            lj_energy = lj_energy + oscillator_scale * osc_energy

        return lj_energy

    def log_prob_single(x: jax.Array):
        return -energy_single(x)

    return JaxEval.create_from_log_prob_single(log_prob_single)
