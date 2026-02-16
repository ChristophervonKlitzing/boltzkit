import numpy as np
from .base import NumPyTarget

from boltzkit.utils.cached_repo import CachedRepo
from boltzkit.utils.molecular.energy_eval import (
    kB_in_eV_per_K,
    ParallelEnergyEval,
    SequentialEnergyEval,
)


from openmm import app
import openmm as mm


class MolecularBoltzmann(NumPyTarget):
    def __init__(
        self,
        path: str,
        n_workers: None | int = -1,
        # energy_transform: FrameworkAgnosticFunction | None = None,
        **kwargs,
    ):
        self._repo = CachedRepo(path, **kwargs)
        self._init_openmm()

        self.temperature = self._repo.config["temperature"]
        self._spatial_dim = 3
        self._n_nodes: int = self.system.getNumParticles()
        dim = self.spatial_dim * self.n_atoms
        super().__init__(dim)

        self._init_energy_eval(n_workers)

    def _init_openmm(self):
        pdb_file = self._repo.config["pdb_file"]
        pdb_file_path = self._repo.load_file(pdb_file)
        forcefield = self._repo.config["forcefield"]
        system_args = self._repo.config.get("system_args", {})

        self.pdb = app.PDBFile(pdb_file_path.absolute().as_posix())
        self.ff = app.ForceField(forcefield)
        self.system: mm.System = self.ff.createSystem(self.pdb.topology, **system_args)

    def _init_energy_eval(self, n_workers: int | None):
        if n_workers is None:
            self._energy_eval = SequentialEnergyEval(self.pdb.topology, self.system)
        elif isinstance(n_workers, int):
            self._energy_eval = ParallelEnergyEval(self.pdb.topology, self.system)
        else:
            raise TypeError

    def _normalize_to_temp(
        self, energy: np.ndarray, temperature: float | None = None
    ) -> np.ndarray:
        T = temperature if temperature is not None else self.temperature
        log_probs = energy / (kB_in_eV_per_K * T)
        return log_probs

    def _numpy_log_prob_and_score(self, x):
        energy, forces = self._energy_eval.evaluate_batch(x)
        log_probs = self._normalize_to_temp(energy)
        scores = self._normalize_to_temp(forces)
        return log_probs, scores

    def _numpy_log_prob(self, x):
        energy, _ = self._energy_eval.evaluate_batch(x, include_forces=False)
        log_probs = self._normalize_to_temp(energy)
        return log_probs

    def _numpy_score(self, x):
        _, forces = self._energy_eval.evaluate_batch(x, include_energy=False)
        scores = self._normalize_to_temp(forces)
        return scores

    @property
    def spatial_dim(self) -> int:
        return self._spatial_dim

    @property
    def n_atoms(self) -> int:
        return self._n_nodes

    def can_sample(self):
        return False


if __name__ == "__main__":
    target = MolecularBoltzmann("datasets/chrklitz99/test_system")
    log_probs = target.get_log_prob(np.random.randn(5, 22, 3))
    print(log_probs)
