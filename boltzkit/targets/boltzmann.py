from .base import NumPyTarget

# from .core.boltzmann import ...


class MolecularBoltzmann(NumPyTarget):
    def __init__(self):
        # Just a wrapper
        # Most of the implementation and dispatching happens in `.core.boltzmann`

        self._spatial_dim = 3
        self._n_nodes = ...
        dim = self.spatial_dim * self.n_atoms

        super().__init__(dim)

    def _numpy_log_prob(self, x):
        raise NotImplementedError

    def _numpy_score(self, x):
        raise NotImplementedError

    @property
    def spatial_dim(self) -> int:
        return self._spatial_dim

    @property
    def n_atoms(self) -> int:
        return self._n_nodes
