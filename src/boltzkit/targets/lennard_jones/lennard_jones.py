from boltzkit.utils.cached_repo import create_cached_repo
from boltzkit.utils.dataloader import CacheLoadingArgs, CachedRepoDatasetLoader

from boltzkit.targets.base import (
    BaseTarget,
    DispatchedDensityProvider,
    ExternalDatasetProvider,
)


class LennardJones(BaseTarget, DispatchedDensityProvider, ExternalDatasetProvider):
    def __init__(
        self,
        n_particles: int,
        spatial_dims: int = 3,
        energy_factor: float = 1.0,
        **kwargs,
    ):
        dim = n_particles * spatial_dims

        super().__init__(dim=dim, **kwargs)

        self._spatial_dims = spatial_dims
        self._n_particles = n_particles
        self._energy_factor = energy_factor

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

    @classmethod
    def create_with_dataset(
        cls,
        n_particles: int,
        cached_repo_uri: str,
        dataset_caching_args: CacheLoadingArgs | None,
        **kwargs,
    ):
        if dataset_caching_args is None:
            dataset_caching_args = {}

        instance = cls(n_particles=n_particles)
        dataset_repo = create_cached_repo(cached_repo_uri, **kwargs)
        dataset_loader = CachedRepoDatasetLoader(
            kB_T=instance.kB_T,
            cached_repo=dataset_repo,
            log_prob_fn=instance.get_log_prob,
            score_fn=instance.get_score,
            caching_args=dataset_caching_args,
        )
        instance.set_dataset_loader(dataset_loader)
        return instance

    @classmethod
    def create_LJ13(
        cls,
        cached_repo_uri: str = "datasets/chrklitz99/lennard_jones_13",
        dataset_caching_args: CacheLoadingArgs | None = None,
        **kwargs,
    ):
        cls.create_with_dataset(
            n_particles=13,
            cached_repo_uri=cached_repo_uri,
            dataset_caching_args=dataset_caching_args,
            **kwargs,
        )

    @classmethod
    def create_LJ13(
        cls,
        cached_repo_uri: str = "datasets/chrklitz99/lennard_jones_55",
        dataset_caching_args: CacheLoadingArgs | None = None,
        **kwargs,
    ):
        cls.create_with_dataset(
            n_particles=55,
            cached_repo_uri=cached_repo_uri,
            dataset_caching_args=dataset_caching_args,
            **kwargs,
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

    lp_new = lj_new.get_log_prob(x_torch)
    print("new", lp_new)

    x_np = x_torch.numpy()
    lp_np = lj.get_log_prob(x_np)
    print(lp_np[:5])

    assert np.allclose(x_torch.numpy(), x_jax)
    assert np.allclose(lp_torch.numpy(), lp_jax, rtol=0.001)
    assert np.allclose(lp_torch, lp_np, rtol=0.001)
    assert np.allclose(score_torch.numpy(), score_jax, rtol=0.001, atol=0.1)
