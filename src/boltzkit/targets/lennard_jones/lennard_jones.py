from boltzkit.targets.base import DispatchedTarget
from boltzkit.utils.cached_repo import CachedRepo, create_cached_repo
from boltzkit.utils.dataset import Dataset


class LennardJones(DispatchedTarget):
    def __init__(
        self,
        n_particles: int,
        spatial_dims: int = 3,
        energy_factor: float = 1.0,
        dataset_repo: CachedRepo | None = None,
    ):
        super().__init__(n_particles * spatial_dims)

        self._spatial_dims = spatial_dims
        self._n_particles = n_particles
        self._energy_factor = energy_factor

        self._dataset_repo = dataset_repo

    @classmethod
    def create_LJ13(
        cls, cached_repo_uri: str = "datasets/chrklitz99/lennard_jones_13", **kwargs
    ):
        return LennardJones(
            n_particles=13, dataset_repo=create_cached_repo(cached_repo_uri, **kwargs)
        )

    @classmethod
    def create_LJ55(
        cls, cached_repo_uri: str = "datasets/chrklitz99/lennard_jones_55", **kwargs
    ):
        return LennardJones(
            n_particles=13, dataset_repo=create_cached_repo(cached_repo_uri, **kwargs)
        )

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
        allow_autogen: bool = True,
        cache_log_probs: bool = True,
        cache_scores: bool = False,
    ) -> Dataset:
        if self._dataset_repo is None:
            raise ValueError(
                "Could not load data due to no data being provided during initialiation of this target"
            )

        return Dataset.create_from_cached_repo(
            self._dataset_repo,
            type=type,
            length=length,
            kB_T=1.0,
            include_samples=include_samples,
            include_log_probs=include_log_probs,
            include_scores=include_scores,
            allow_autogen=allow_autogen,
            cache_log_probs=cache_log_probs,
            cache_scores=cache_scores,
            log_prob_fn=self.get_log_prob,
            score_fn=self.get_score,
        )

    def _create_numpy_eval(self):
        from ._np_lj import NumpyLennardJonesEval

        return NumpyLennardJonesEval(
            n_particles=self._n_particles,
            spatial_dims=self._spatial_dims,
            energy_factor=self._energy_factor,
        )

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

    print(lp_torch.numpy()[:5])
    # print(score_torch)

    x_jax = jax.numpy.array(x_torch.numpy())
    lp_jax = lj.get_log_prob(x_jax)
    score_jax = lj.get_score(x_jax)
    print(lp_jax[:5])
    # print(score_jax)

    x_np = x_torch.numpy()
    lp_np = lj.get_log_prob(x_np)
    print(lp_np[:5])

    assert np.allclose(x_torch.numpy(), x_jax)
    assert np.allclose(lp_torch.numpy(), lp_jax, rtol=0.001)
    assert np.allclose(lp_torch, lp_np, rtol=0.001)
    assert np.allclose(score_torch.numpy(), score_jax, rtol=0.001, atol=0.1)
