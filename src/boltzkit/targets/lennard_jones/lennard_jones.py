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
        raise NotImplementedError


if __name__ == "__main__":
    import torch

    torch.random.manual_seed(0)

    lj = LennardJones(n_particles=13)

    x = torch.randn((2, lj.dim))
    lp1 = lj.get_log_prob(x)
    score1 = lj.get_score(x)

    print(lp1)
    print(score1)
