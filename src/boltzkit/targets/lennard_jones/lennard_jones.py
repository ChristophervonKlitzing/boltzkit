from boltzkit.targets.base import DispatchedTarget
from boltzkit.utils.dataset import Dataset


class LennardJones(DispatchedTarget):
    def __init__(
        self, n_particles: int, spatial_dims: int = 3, energy_factor: float = 1.0
    ):
        super().__init__(n_particles * spatial_dims)

        self._spatial_dims = spatial_dims
        self._n_particles = n_particles
        self._energy_factor = energy_factor

    def can_sample(self):
        return False

    def load_dataset(
        self,
        type,
        length,
        *,
        include_samples=True,
        include_log_probs=False,
        include_scores=False,
    ) -> Dataset:
        raise NotImplementedError

    def _create_numpy_eval(self):
        raise NotImplementedError

    def _create_torch_eval(self):
        from ._torch_lj import TorchLennardJonesEval

        return TorchLennardJonesEval(
            n_particles=self._n_particles,
            spatial_dims=self._spatial_dims,
            energy_factor=self._energy_factor,
        )

    def _create_jax_eval(self):
        from ._jax_lj import create_jax_lennard_jones_eval

        return create_jax_lennard_jones_eval(
            n_particles=self._n_particles,
            spatial_dims=self._spatial_dims,
            energy_factor=self._energy_factor,
        )


if __name__ == "__main__":
    import torch
    import jax
    import numpy as np

    torch.random.manual_seed(0)

    lj = LennardJones(n_particles=13)

    x_torch = torch.randn((1000, lj.dim))
    lp_torch = lj.get_log_prob(x_torch)
    score_torch = lj.get_score(x_torch)

    print(lp_torch.numpy())
    # print(score_torch)

    x_jax = jax.numpy.array(x_torch.numpy())
    lp_jax = lj.get_log_prob(x_jax)
    score_jax = lj.get_score(x_jax)
    print(lp_jax)
    # print(score_jax)

    print(x_torch.dtype, x_jax.dtype)

    assert np.allclose(x_torch.numpy(), x_jax)
    assert np.allclose(lp_torch.numpy(), lp_jax, rtol=0.001)
    assert np.allclose(score_torch.numpy(), score_jax, rtol=0.001, atol=0.1)
