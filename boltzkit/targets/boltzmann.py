from typing import Literal

import numpy as np

from boltzkit.utils.molecular.conversion import vec3_list_to_numpy


from boltzkit.targets.base import NumPyTarget

from boltzkit.utils.cached_repo import CachedRepo
from boltzkit.utils.molecular.energy_eval import (
    kB_in_eV_per_K,
    ParallelEnergyEval,
    SequentialEnergyEval,
)


from openmm import app
import openmm as mm
from openmm import unit

from boltzkit.utils.molecular.tica import TicaModelWithLengthScale
from boltzkit.utils.molecular.z_matrix_factory import ZMatrixFactory
import mdtraj as md


class MolecularBoltzmann(NumPyTarget):
    def __init__(
        self,
        path: str,
        n_workers: None | int = -1,
        length_unit: Literal["angstrom", "nanometer"] | float = "nanometer",
        # energy_transform: FrameworkAgnosticFunction | None = None,
        **kwargs,
    ):
        self._repo = CachedRepo(path, **kwargs)
        self._init_openmm()

        self._temperature = self._repo.config.get("temperature", 300.0)
        self._spatial_dim = 3
        self._n_nodes: int = self._system.getNumParticles()
        dim = self.spatial_dim * self.n_atoms
        super().__init__(dim)

        self._n_workers = n_workers
        self._energy_eval = None

        self._pos_min_energy_cache = None

        if isinstance(length_unit, str):
            if length_unit == "angstrom":
                self._length_scale = 0.1
            elif length_unit == "nanometer":
                self._length_scale = 1.0
        else:
            self._length_scale = float(length_unit)

    def _init_openmm(self):
        pdb_key = "pdb_file"
        pdb_file = self._repo.config.get(pdb_key, None)
        if pdb_file is None:
            # automatic search for pdb file (there must exist exactly one for automatic search)
            pdb_file_list = self._repo.find_file(r".*\.pdb$")
            if len(pdb_file_list) != 1:
                raise ValueError(
                    f"Expected exactly one .pdb file in the repository, "
                    f"but found {len(pdb_file_list)}. "
                    f"Please specify the main PDB file explicitly in the config "
                    f"using '{pdb_key}'."
                )
            pdb_file = pdb_file_list[0]
            print(
                f"Key '{pdb_key}' not specified. Use automatically detected .pdb file '{pdb_file}'."
            )

        pdb_file_path = self._repo.load_file(pdb_file)
        forcefields = self._repo.config.get(
            "forcefields", ["amber99sbildn.xml", "amber99_obc.xml"]
        )
        system_args = self._repo.config.get("system_args", {})

        # Create system
        self._pdb = app.PDBFile(pdb_file_path.absolute().as_posix())
        self._forcefield = app.ForceField(*forcefields)
        self._system: mm.System = self._forcefield.createSystem(
            self._pdb.topology, **system_args
        )

    def get_openmm_topology(self):
        return self._pdb.topology

    def get_openmm_system(self):
        return self._system

    def get_mdtraj_topology(self):
        return md.Topology.from_openmm(self.get_openmm_topology())

    def _init_energy_eval(self, n_workers: int | None):
        if n_workers is None:
            self._energy_eval = SequentialEnergyEval(self._pdb.topology, self._system)
        elif isinstance(n_workers, int):
            self._energy_eval = ParallelEnergyEval(
                self._pdb.topology, self._system, n_workers=n_workers
            )
        else:
            raise TypeError

    @property
    def energy_eval(self):
        if self._energy_eval is None:
            self._init_energy_eval(self._n_workers)
        return self._energy_eval

    def _energy_to_log_prob(
        self, energy: np.ndarray, temperature: float | None = None
    ) -> np.ndarray:
        T = temperature if temperature is not None else self._temperature
        log_probs = -energy / (kB_in_eV_per_K * T)
        return log_probs

    def _forces_to_score(
        self, forces: np.ndarray, temperature: float | None = None
    ) -> np.ndarray:
        T = temperature if temperature is not None else self._temperature
        score = forces / (kB_in_eV_per_K * T)
        return score

    def _numpy_log_prob_and_score(self, x):
        x_nm = x * self._length_scale
        energy, forces_nm = self.energy_eval.evaluate_batch(x_nm)
        forces = forces_nm * self._length_scale
        log_probs = self._energy_to_log_prob(energy)
        scores = self._forces_to_score(forces)
        return log_probs, scores

    def _numpy_log_prob(self, x):
        x_nm = x * self._length_scale
        energy, _ = self.energy_eval.evaluate_batch(x_nm, include_forces=False)
        log_probs = self._energy_to_log_prob(energy)
        return log_probs

    def _numpy_score(self, x):
        x_nm = x * self._length_scale
        _, forces_nm = self.energy_eval.evaluate_batch(x_nm, include_energy=False)
        forces = forces_nm * self._length_scale
        scores = self._forces_to_score(forces)
        return scores

    def get_energy_and_forces_numpy(self, x: np.ndarray):
        x_nm = x * self._length_scale
        energy, forces_nm = self.energy_eval.evaluate_batch(x_nm)
        forces = forces_nm * self._length_scale
        return energy, forces

    def _numpy_energy(self, x):
        x_nm = x * self._length_scale
        energy, _ = self.energy_eval.evaluate_batch(x_nm, include_forces=False)
        return energy

    def _numpy_forces(self, x):
        x_nm = x * self._length_scale
        _, forces_nm = self.energy_eval.evaluate_batch(x_nm, include_energy=False)
        forces = forces_nm * self._length_scale
        return forces

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
            mdtraj_topology = md.Topology.from_openmm(self._pdb.topology)
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
        simulation = app.Simulation(self._pdb.topology, self._system, integrator)
        simulation.context.setPositions(self._pdb.positions)

        # optimization is deterministic
        simulation.minimizeEnergy()
        state: mm.State = simulation.context.getState(getPositions=True)
        minimized_positions = state.getPositions()
        minimized_positions = vec3_list_to_numpy(minimized_positions)
        return minimized_positions.reshape((self.dim,))

    def get_position_min_energy(self) -> np.ndarray:
        pos_nm = None
        if self._pos_min_energy_cache is not None:
            pos_nm = self._pos_min_energy_cache
        else:
            remote_path = self._repo.config.get("position_min_energy", None)
            if remote_path is not None:
                local_path = self._repo.load_file(remote_path)
                pos_min_energy: np.ndarray = np.load(local_path)
                self._pos_min_energy_cache = pos_min_energy.reshape((self.dim,))
                pos_nm = self._pos_min_energy_cache

        if pos_nm is None:
            import warnings

            warnings.warn(
                "Could not find minimum energy position and use automatically determined position instead"
            )
            # Position min energy not specifed -> determine automatically
            pos_nm = self._compute_position_min_energy()
            self._pos_min_energy_cache = pos_nm

        return pos_nm / self._length_scale

    def get_tica_model(self):
        tica_key = "tica"
        tica_remote_path = self._repo.config.get(tica_key, None)
        if tica_remote_path is None:
            tica_file_list = self._repo.find_file(r"^tica.pkl$")
            if len(tica_file_list) != 1:
                raise ValueError(
                    f"Expected exactly one tica file in the repository, "
                    f"but found {len(tica_file_list)}. "
                    f"Please specify the main tica file explicitly in the config "
                    f"using '{tica_key}'."
                )
            tica_remote_path = tica_file_list[0]

        tica_local_path = self._repo.load_file(tica_remote_path)
        return TicaModelWithLengthScale.from_path(tica_local_path, self._length_scale)

    def load_dataset(
        self, T: float | str | int, type: Literal["train", "val", "test"]
    ) -> np.ndarray | None:
        datasets: dict[str, dict[str, str]] | None = self._repo.config.get(
            "datasets", None
        )
        if datasets is None:
            return None

        if isinstance(T, int):
            T = float(T)

        temp_cfg = datasets.get(str(T), None)
        if temp_cfg is None:
            return None

        dataset_remote_fname = temp_cfg.get(type, None)
        if dataset_remote_fname is None:
            return None

        local_fname = self._repo.load_file(dataset_remote_fname)
        return np.load(local_fname) / self._length_scale

    @property
    def spatial_dim(self) -> int:
        return self._spatial_dim

    @property
    def n_atoms(self) -> int:
        return self._n_nodes

    def can_sample(self):
        return False


def print_z_matrix(z_matrix: list[tuple[int, int, int, int]]):
    if not z_matrix:
        print("--- z-matrix ---")
        print("(empty)")
        print("----------------")
        return

    # Transpose to compute column widths
    columns = list(zip(*z_matrix))
    col_widths = [max(len(str(v)) for v in col) for col in columns]

    # Format rows first so we know total width
    formatted_rows = [
        "  ".join(f"{value:>{col_widths[i]}}" for i, value in enumerate(row))
        for row in z_matrix
    ]

    row_width = max(len(row) for row in formatted_rows)

    title = "--- z-matrix ---"
    if len(title) < row_width:
        title = title.center(row_width, "-")

    print(title)
    for row in formatted_rows:
        print(row)
    print("-" * row_width)


if __name__ == "__main__":
    target = MolecularBoltzmann(
        "datasets/chrklitz99/test_system", length_unit="angstrom"
    )
    # log_probs = target.get_log_prob(np.random.randn(5, 22, 3))
    # print(log_probs)

    pos_min_energy = target.get_position_min_energy()
    # print(pos_min_energy)

    print(target.load_dataset(300, "val")[0])
