import numpy as np

from boltzkit.utils.molecular.conversion import vec3_list_to_numpy


from .base import NumPyTarget

from boltzkit.utils.cached_repo import CachedRepo
from boltzkit.utils.molecular.energy_eval import (
    kB_in_eV_per_K,
    ParallelEnergyEval,
    SequentialEnergyEval,
)


from openmm import app
import openmm as mm
from openmm import unit

from boltzkit.utils.molecular.z_matrix_factory import ZMatrixFactory
import mdtraj as md


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

        self.temperature = self._repo.config.get("temperature", 300.0)
        self._spatial_dim = 3
        self._n_nodes: int = self.system.getNumParticles()
        dim = self.spatial_dim * self.n_atoms
        super().__init__(dim)

        self._init_energy_eval(n_workers)

        self._pos_min_energy_cache = None

    def _init_openmm(self):
        pdb_file = self._repo.config["pdb_file"]
        pdb_file_path = self._repo.load_file(pdb_file)
        forcefield = self._repo.config.get("forcefield", "amber14-all.xml")
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

    def _energy_to_log_prob(
        self, energy: np.ndarray, temperature: float | None = None
    ) -> np.ndarray:
        T = temperature if temperature is not None else self.temperature
        log_probs = -energy / (kB_in_eV_per_K * T)
        return log_probs

    def _forces_to_score(
        self, forces: np.ndarray, temperature: float | None = None
    ) -> np.ndarray:
        T = temperature if temperature is not None else self.temperature
        score = forces / (kB_in_eV_per_K * T)
        return score

    def _numpy_log_prob_and_score(self, x):
        energy, forces = self._energy_eval.evaluate_batch(x)
        log_probs = self._energy_to_log_prob(energy)
        scores = self._forces_to_score(forces)
        return log_probs, scores

    def _numpy_log_prob(self, x):
        energy, _ = self._energy_eval.evaluate_batch(x, include_forces=False)
        log_probs = self._energy_to_log_prob(energy)
        return log_probs

    def _numpy_score(self, x):
        _, forces = self._energy_eval.evaluate_batch(x, include_energy=False)
        scores = self._forces_to_score(forces)
        return scores

    def _get_cartesian_indices(self, z_matrix: list[tuple[int, int, int, int]]):
        """
        Auto-determines the atom indices, which are referenced but not present in the
        """

    def get_z_matrix(self, allow_autogen=True) -> list[tuple[int, int, int, int]]:
        """
        Generate or get a z-matrix for this system.

        :param allow_autogen: Whether to automatically generate a z-matrix if none is specifed in the system config.
        :type allow_autogen: bool
        """
        z_matrix: None | list[tuple[int, int, int, int]] = self._repo.config.get(
            "z_matrix", None
        )

        if z_matrix is None and allow_autogen:
            print("Create z-matrix using ZMatrixFactory")
            mdtraj_topology = md.Topology.from_openmm(self.pdb.topology)
            factory = ZMatrixFactory(mdtraj_topology)
            np_z_matrix = factory.build_with_templates()[0]
            z_matrix = np_z_matrix.tolist()

        if z_matrix is None:
            raise ValueError(
                "System has no pre-specified z-matrix and `autogen` is set to False"
            )

        return z_matrix

    def _compute_position_min_energy(self):
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        simulation = app.Simulation(self.pdb.topology, self.system, integrator)
        simulation.context.setPositions(self.pdb.positions)

        # optimization is deterministic
        simulation.minimizeEnergy()
        state: mm.State = simulation.context.getState(getPositions=True)
        minimized_positions = state.getPositions()
        minimized_positions = vec3_list_to_numpy(minimized_positions)
        return minimized_positions.reshape((self.dim,))

    def get_position_min_energy(self) -> np.ndarray:
        if self._pos_min_energy_cache is not None:
            return self._pos_min_energy_cache

        remote_path = self._repo.config.get("position_min_energy", None)
        if remote_path is not None:
            local_path = self._repo.load_file(remote_path)
            pos_min_energy: np.ndarray = np.load(local_path)
            self._pos_min_energy_cache = pos_min_energy.reshape((self.dim,))
            return self._pos_min_energy_cache

        import warnings

        warnings.warn(
            "Could not find minimum energy position and use automatically determined position instead"
        )
        # Position min energy not specifed -> determine automatically
        pos_min_energy = self._compute_position_min_energy()
        self._pos_min_energy_cache = pos_min_energy
        return self._pos_min_energy_cache

    def create_internal_coordinate_trafo(self):
        self.coordinate_trafo_openmm = bg.GlobalInternalCoordinateTransformation(
            z_matrix=z_matrix,
            enforce_boundaries=True,
            normalize_angles=True,
        )

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
    z_matrix = target.get_z_matrix()
    print(z_matrix)

    pos_min_energy = target.get_position_min_energy()
    print(pos_min_energy.shape)
