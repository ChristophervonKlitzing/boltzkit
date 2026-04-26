import os
from typing import Literal
import warnings

import numpy as np

from boltzkit.utils.molecular.conversion import vec3_list_to_numpy


from boltzkit.targets.base import NumPyTarget

from boltzkit.utils.cached_repo import CachedRepo, VirtualRepo, create_cached_repo
from boltzkit.utils.dataset import Dataset
from boltzkit.utils.molecular.energy_eval import (
    kB_in_eV_per_K,
    ParallelEnergyEval,
    SequentialEnergyEval,
)

from openmm import app
import openmm as mm
from openmm import unit

from boltzkit.utils.molecular.pdbpatch import fixed_atom_names
from boltzkit.utils.molecular.tica import TicaModelWithLengthScale
from boltzkit.utils.molecular.z_matrix_factory import ZMatrixFactory
import mdtraj as md

_CHARMM_KEY = "charmm_files"
_FORCEFIELDS_KEY = "forcefields"
_PDB_KEY = "pdb_file"
_PSF_KEY = "psf_file"


def _parse_system_args(system_args: dict):
    system_args = system_args.copy()

    # Parse implicitSolvent if present
    implicit_solvent_key = "implicitSolvent"
    if implicit_solvent_key in system_args:
        solvent = system_args[implicit_solvent_key]
        if solvent == "OBC1":
            system_args[implicit_solvent_key] = app.OBC1
        elif solvent == "OBC2":
            system_args[implicit_solvent_key] = app.OBC2
        else:
            raise ValueError(
                f"Unsupported value for key '{implicit_solvent_key}': Got '{solvent}' but expected either 'OBC1' or 'OBC2'"
            )

    # Parse hydrogenMass if present
    hydrogen_mass_key = "hydrogenMass"
    if hydrogen_mass_key in system_args:
        factor = system_args[hydrogen_mass_key]
        if not (isinstance(factor, float) and factor > 0):
            raise ValueError(
                f"Unsupported value for key '{hydrogen_mass_key}': Got '{factor}' but expected it to be a positive float"
            )
        system_args[hydrogen_mass_key] = factor * unit.amu

    return system_args


class MolecularBoltzmann(NumPyTarget):
    """
    awd
    """

    def __init__(
        self,
        path: str | CachedRepo,
        *,
        n_workers: None | int = -1,
        openmm_platform: Literal["CPU", "CUDA"] | None = "CPU",
        length_unit: Literal["angstrom", "nanometer"] | float = "nanometer",
        # energy_transform: FrameworkAgnosticFunction | None = None,
        **kwargs,
    ):
        """
        Wraps a molecular system for energy/log-density and force/score evaluation using OpenMM.

        This class handles conversion of positions to the appropriate length scale
        and sets up the OpenMM System from the given repository path, e.g., a Hugging Face
        repository such as `datasets/chrklitz99/alanine_dipeptide`.

        ### Number of workers and OpenMM platform

        There are effectively two sensible modes for evaluation:

        1. **CPU mode**: `openmm_platform="CPU"` and `n_workers=-1` or a positive integer
        (performs batch evaluation in parallel across multiple processes).

        2. **GPU mode**: `openmm_platform="CUDA"` and `n_workers=None`
        (performs sequential batch evaluation on a single GPU).

        If training itself is parallelized across multiple GPUs, mode 2 can be appropriate,
        since sequential evaluation on each GPU may be faster than parallel evaluation across
        multiple CPUs.

        Parameters
        ----------
        path : str
            Path to the repository that configures the system or a cached repo instance.
            In the latter case, no new cached repo is initialized but the provided one is used.
        n_workers : int or None, optional, default=-1
            Number of parallel workers for computations. -1 uses all available CPU cores,
            None means sequential evaluation. Applies to all platforms, but using a GPU
            (CUDA) with multiple workers triggers a warning because parallel evaluation
            makes little sense on a single GPU.
        openmm_platform : {"CPU", "CUDA"} or None, optional, default="CPU"
            OpenMM computation platform to use. None lets OpenMM select CUDA if available,
            with a fallback to CPU. Parallel evaluation (`n_workers`) always applies, but
            the combinations CUDA + multiple workers will issue a warning.
        length_unit : {"angstrom", "nanometer"} or float, optional, default="nanometer"
            Scaling factor for positional units. If a string, must be "angstrom" or "nanometer".
            If a float, interpreted as a custom scale.
        **kwargs : dict
            Additional arguments passed to the repository loader.
        """
        if openmm_platform != "CPU" and n_workers is not None:
            warnings.warn(
                f"Parallel energy & force evaluation ({n_workers=}) makes no sense when not using {openmm_platform=}"
            )

        if isinstance(path, CachedRepo):
            self._repo = path
        elif isinstance(path, str):
            self._repo = create_cached_repo(path, **kwargs)
        else:
            raise ValueError(
                f"The provided 'path' must be either of type '{CachedRepo.__name__}' or '{str.__name__}' but got '{type(path).__name__}'"
            )

        self._init_openmm()

        self._temperature = self._repo.config.get("temperature", 300.0)
        self._spatial_dim = 3
        self._n_nodes: int = self._system.getNumParticles()
        dim = self.spatial_dim * self.n_atoms
        super().__init__(dim)

        self._n_workers = n_workers
        self._openmm_platform = openmm_platform
        self._energy_eval = None

        self._pos_min_energy_cache = None

        if isinstance(length_unit, str):
            if length_unit == "angstrom":
                self._length_scale = 0.1
            elif length_unit == "nanometer":
                self._length_scale = 1.0
        else:
            self._length_scale = float(length_unit)

    @classmethod
    def create_from_pdb(
        cls,
        name: str,
        pdb_fpath: str,
        *,
        forcefields: list[str] | tuple[str, ...] = (
            "amber99sbildn.xml",
            "amber99_obc.xml",
        ),
        temperature: float = 300.0,
        cached_repo_args: dict | None = None,
        boltzmann_args: dict | None = None,
    ):
        if not os.path.exists(pdb_fpath) and os.path.isfile(pdb_fpath):
            raise ValueError(
                f"The given 'pdb_fpath' string is not pointing to a file: '{pdb_fpath}'"
            )

        forcefield_str = "\n".join([f"  - {ff}" for ff in forcefields])

        info_yaml = (
            f"temperature: {temperature} \n"
            "pdb_file: topology.pdb \n"
            "forcefields: \n"
            f"{forcefield_str}"
        )

        import shutil

        pdb_copy_action = lambda dest_fpath: shutil.copy(pdb_fpath, dest_fpath)

        file_tree = {"info.yaml": info_yaml, "topology.pdb": pdb_copy_action}

        if cached_repo_args is None:
            cached_repo_args = {}

        virt_repo = create_cached_repo(
            f"virtual://{name}", file_tree=file_tree, **cached_repo_args
        )

        if boltzmann_args is None:
            boltzmann_args = {}

        return MolecularBoltzmann(virt_repo, **boltzmann_args)

    def _get_system_args(self) -> dict:
        system_args = self._repo.config.get("system_args", {})
        return _parse_system_args(system_args)

    def _find_pdb_file_path(self) -> str:
        pdb_file = self._repo.config.get(_PDB_KEY, None)
        if pdb_file is None:
            # automatic search for pdb file (there must exist exactly one for automatic search)
            pdb_file_list = self._repo.find_file(r".*\.pdb$")
            if len(pdb_file_list) != 1:
                raise ValueError(
                    f"Expected exactly one .pdb file in the repository, "
                    f"but found {len(pdb_file_list)}. "
                    f"Please specify the main PDB file explicitly in the config "
                    f"using '{_PDB_KEY}'."
                )
            pdb_file = pdb_file_list[0]
            print(
                f"Key '{_PDB_KEY}' not specified. Use automatically detected .pdb file '{pdb_file}'."
            )

        pdb_file_path = self._repo.load_file(pdb_file)
        return pdb_file_path.absolute().as_posix()

    def _init_openmm(self):
        if _CHARMM_KEY in self._repo.config:
            pdb, system = self._init_openmm_from_CHARMM_params()
        elif _FORCEFIELDS_KEY in self._repo.config:
            pdb, system = self._init_openmm_from_forcefields()
        else:
            raise ValueError(
                f"Unsupported config format: Use either '{_CHARMM_KEY}' or '{_FORCEFIELDS_KEY}' to configure the system"
            )

        self._system = system
        self._pdb = pdb

    def _init_openmm_from_forcefields(self):
        forcefields = self._repo.config.get(
            _FORCEFIELDS_KEY, ["amber99sbildn.xml", "amber99_obc.xml"]
        )

        pdb = app.PDBFile(self._find_pdb_file_path())
        ff = app.ForceField(*forcefields)
        system_args = self._get_system_args()
        system: mm.System = ff.createSystem(
            pdb.topology, nonbondedMethod=app.NoCutoff, **system_args
        )
        return pdb, system

    def _init_openmm_from_CHARMM_params(self):
        with fixed_atom_names(TYR=["HT1", "HT2", "HT3"]):
            # Contains the topology
            psf_file = self._repo.load_file(self._repo.config[_PSF_KEY]).as_posix()
            psf = app.CharmmPsfFile(psf_file)

            # Contains the structure (e.g. positions)
            pdb = app.PDBFile(self._find_pdb_file_path())

        system_args = self._get_system_args()

        params_files = self._repo.config.get(_CHARMM_KEY, [])
        params_paths = [
            self._repo.load_file(pfile).as_posix() for pfile in params_files
        ]
        params = app.CharmmParameterSet(*params_paths)

        system: mm.System = psf.createSystem(
            params,
            nonbondedMethod=app.NoCutoff,
            **system_args,
        )
        return pdb, system

    def get_openmm_topology(self):
        return self._pdb.topology

    def get_openmm_system(self):
        return self._system

    def get_mdtraj_topology(self):
        return md.Topology.from_openmm(self.get_openmm_topology())

    def _init_energy_eval(self, n_workers: int | None):
        if n_workers is None:
            self._energy_eval = SequentialEnergyEval(
                self._pdb.topology, self._system, platform=self._openmm_platform
            )
        elif isinstance(n_workers, int):
            self._energy_eval = ParallelEnergyEval(
                self._pdb.topology,
                self._system,
                n_workers=n_workers,
                platform=self._openmm_platform,
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

    def get_energy_and_forces(
        self, x: np.ndarray, include_energy: bool = True, include_forces: bool = True
    ):
        x_nm = x * self._length_scale
        energy, forces_nm = self.energy_eval.evaluate_batch(
            x_nm, include_energy=include_energy, include_forces=include_forces
        )
        if forces_nm is not None:
            forces = forces_nm * self._length_scale
        else:
            forces = None
        return energy, forces

    def _numpy_log_prob_and_score(self, x):
        energy, forces = self.get_energy_and_forces(
            x, include_energy=True, include_forces=True
        )
        log_probs = self._energy_to_log_prob(energy)
        scores = self._forces_to_score(forces)
        return log_probs, scores

    def _numpy_log_prob(self, x):
        energy, _ = self.get_energy_and_forces(
            x, include_energy=True, include_forces=False
        )
        log_probs = self._energy_to_log_prob(energy)
        return log_probs

    def _numpy_score(self, x):
        _, forces = self.get_energy_and_forces(
            x, include_energy=False, include_forces=True
        )
        scores = self._forces_to_score(forces)
        return scores

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
            tica_file_list = self._repo.find_file(r"^tica.*\.pkl$")
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

    @property
    def spatial_dim(self) -> int:
        return self._spatial_dim

    @property
    def n_atoms(self) -> int:
        return self._n_nodes

    def can_sample(self):
        return False

    def load_dataset(
        self,
        type: Literal["train", "val", "test"],
        length: int = -1,
        *,
        T: float | int | None = None,
        #
        include_samples: bool = True,
        include_energies: bool = False,
        include_forces: bool = False,
        #
        cache_energies: bool = True,
        cache_forces: bool = False,
        #
        allow_autogen: bool = True,
    ) -> Dataset:
        """
        Load a dataset for a given temperature and dataset split.

        This method retrieves samples and optionally associated
        energies and forces. It supports loading
        precomputed data, retrieving cached values, and automatically
        generating missing quantities (energies or forces) on demand.

        Parameters
        ----------
        type : Literal["train", "val", "test"]
            Dataset split to load.
        length : int, optional
            Maximum number of samples to load. If -1, all available samples are used.
        T : float | int | None
            Temperature (in Kelvin) identifying the dataset. Integers are cast to float. If None, the target's temperature is used.
        include_samples : bool, default=True
            Whether to return samples.
        include_energies : bool, default=False
            Whether to include energy values for each sample. Fails if no energies are available and `allow_autogen` is False.
        include_forces : bool, default=False
            Whether to include force values for each sample. Fails if no forces are available and `allow_autogen` is False.
        cache_energies : bool, default=True
            Whether to use cached and/or cache computed energies locally when they are generated. Requires `allow_autogen` to be True.
        cache_forces : bool, default=False
            Whether to use cached and/or cache computed forces locally when they are generated. Requires `allow_autogen` to be True.
        allow_autogen : bool, default=True
            If True, missing energies or forces are computed on demand. Without caching being enabled,
            energies and forces will be re-computed each time this function is called.

        Returns
        -------
        Dataset

        Raises
        ------
        RuntimeError
            If dataset configuration is missing or if the requested temperature
            or dataset split is not found.

        Notes
        -----
        - If energies or forces are missing, they may be:
          1. Loaded from remote files if available,
          2. Retrieved from a local cache if enabled,
          3. Computed on demand if `allow_autogen=True`.
        - Forces are internally scaled by `self._length_scale` when loaded and cached in nanometers.
        """
        datasets: dict[str, dict[str, str]] | None = self._repo.config.get(
            "datasets", None
        )
        if datasets is None:
            raise RuntimeError("Missing datasets config")

        if isinstance(T, int):
            T = float(T)

        temp_cfg = datasets.get(str(T), None)
        if temp_cfg is None:
            available_temps = list(datasets.keys())
            raise RuntimeError(
                f"Missing dataset: "
                f"Searched for temperature {T}K, but only found {available_temps}."
            )

        dset_cfg: dict[str, str] | str | None = temp_cfg.get(type, None)
        if dset_cfg is None:
            available_keys = list(temp_cfg.keys())
            raise RuntimeError(
                f"Missing dataset type for temperature {T}K. "
                f"Searched for type '{type}', but only found {available_keys}."
            )

        if isinstance(dset_cfg, str):
            dset_cfg = {"samples": dset_cfg}

        samples = self.__load_samples(dset_cfg, T, length)

        cache_prefix = f"_cached_dataset_{T}K_{type}"
        energies, forces = self.__load_compute_energies_and_forces(
            samples,
            dset_cfg=dset_cfg,
            autogen=allow_autogen,
            include_energies=include_energies,
            include_forces=include_forces,
            cache_prefix=cache_prefix,
            cache_energies=cache_energies,
            cache_forces=cache_forces,
        )

        if not include_samples:
            samples = None

        kB_T = kB_in_eV_per_K * T
        return Dataset(kB_T, samples=samples, energies=energies, forces=forces)

    def __load_samples(self, dset_cfg: dict[str, str], T: float, length: int):
        samples_remote_fpath = dset_cfg.get("samples", None)
        if samples_remote_fpath is None:
            raise RuntimeError(
                f"Missing 'samples' key for dataset of type '{type}' at temperature {T}K"
            )
        samples_local_fpath = self._repo.load_file(samples_remote_fpath)
        samples_nm: np.ndarray = np.load(samples_local_fpath, mmap_mode="r")

        if length != -1:
            assert samples_nm.shape[0] >= length
            samples_nm = samples_nm[:length]

        samples = samples_nm / self._length_scale
        return samples

    def __load_compute_energies_and_forces(
        self,
        samples: np.ndarray,
        dset_cfg: dict[str, str],
        autogen: bool,
        #
        include_energies: bool,
        include_forces: bool,
        #
        cache_prefix: str,
        cache_energies: bool,
        cache_forces: bool,
    ):
        kv_store = self._repo.get_cached_key_value_store()
        energies_cache_key = cache_prefix + f"_energies"
        forces_cache_key = cache_prefix + f"_forces"
        local_cache_dir = self._repo.local_path

        # Get local energies path
        remote_energies_path = dset_cfg.get("energies", None)
        if remote_energies_path is not None:
            energies_local_fpath = self._repo.load_file(remote_energies_path)
        elif cache_energies and autogen:
            rel = kv_store.get(energies_cache_key)
            energies_local_fpath = local_cache_dir / rel if rel is not None else None
        else:
            energies_local_fpath = None

        # Get local forces path
        remote_forces_path = dset_cfg.get("forces", None)
        if remote_forces_path is not None:
            forces_local_fpath = self._repo.load_file(remote_forces_path)
        elif cache_forces and autogen:
            rel = kv_store.get(forces_cache_key)
            forces_local_fpath = local_cache_dir / rel if rel is not None else None
        else:
            forces_local_fpath = None

        energies = None
        forces = None

        # Load energies and forces from file if possible
        if include_energies and energies_local_fpath is not None:
            print(f"Use pre-computed energies from '{energies_local_fpath}'")
            energies: np.ndarray = np.load(energies_local_fpath)
            if energies.shape[0] > samples.shape[0]:
                energies = energies[: samples.shape[0]]

        if include_forces and forces_local_fpath is not None:
            print(f"Use pre-computed forces from '{forces_local_fpath}'")
            forces_nm: np.ndarray = np.load(forces_local_fpath)
            if forces_nm.shape[0] > samples.shape[0]:
                forces_nm = forces_nm[: samples.shape[0]]
            forces = forces_nm * self._length_scale

        # If necessary and autogen is True, compute energies/forces online
        if autogen:
            energies, forces = self._fill_missing_energies_and_forces(
                samples,
                energies,
                forces,
                include_energies=include_energies,
                include_forces=include_forces,
            )

            # If caching is enabled, cache the computed energies/forces
            if cache_energies and energies is not None:
                relative_energies_fpath = f"{cache_prefix}_energies.npy"
                kv_store.set(energies_cache_key, relative_energies_fpath)
                energies_fpath = local_cache_dir / relative_energies_fpath
                np.save(energies_fpath, energies)
            if cache_forces and forces is not None:
                forces_nm = forces / self._length_scale
                relative_forces_fpath = f"{cache_prefix}_forces.npy"
                kv_store.set(forces_cache_key, relative_forces_fpath)
                forces_fpath = local_cache_dir / relative_forces_fpath
                np.save(forces_fpath, forces_nm)

        return energies, forces

    def _fill_missing_energies_and_forces(
        self,
        samples: np.ndarray,
        energies: np.ndarray | None,
        forces: np.ndarray | None,
        include_energies: bool,
        include_forces: bool,
    ):
        n_samples = samples.shape[0]
        n_energies = 0 if energies is None else energies.shape[0]
        n_forces = 0 if forces is None else forces.shape[0]

        require_energies = include_energies and n_energies < n_samples
        require_forces = include_forces and n_forces < n_samples

        # If there is a mismatch between n_energies and n_forces, first compute the
        # remaining forces or energies to catch up.
        if require_energies and require_forces:
            if n_energies < n_forces:
                energy_gap, _ = self.get_energy_and_forces(
                    samples[n_energies:n_forces],
                    include_energy=True,
                    include_forces=False,
                )
                if energies is None:
                    energies = energy_gap
                else:
                    energies = np.concatenate([energies, energy_gap], 0)
                n_energies += energy_gap.shape[0]
            elif n_forces < n_energies:
                _, forces_gap = self.get_energy_and_forces(
                    samples[n_forces:n_energies],
                    include_energy=False,
                    include_forces=True,
                )
                if forces is None:
                    forces = forces_gap
                else:
                    forces = np.concatenate([forces, forces_gap], 0)
                n_forces += forces_gap.shape[0]

        # Re-compute flags with updated energy/force count
        require_energies = include_energies and n_energies < n_samples
        require_forces = include_forces and n_forces < n_samples

        if require_energies and require_forces:
            assert n_energies == n_forces
            start_idx = n_energies  # or = n_forces
        elif require_energies:
            start_idx = n_energies
        elif require_forces:
            start_idx = n_forces
        else:  # both False
            start_idx = n_samples

        if start_idx < n_samples:  # compute remaining energies and/or forces
            n_missing_energies = 0 if not require_energies else n_samples - n_energies
            n_missing_forces = 0 if not require_forces else n_samples - n_forces
            print(
                f"Compute {n_missing_energies} energies and {n_missing_forces} forces"
            )

            missing_energies, missing_forces = self.get_energy_and_forces(
                samples[start_idx:n_samples],
                include_energy=require_energies,
                include_forces=require_forces,
            )

            if missing_energies is not None:
                if energies is None:
                    energies = missing_energies
                else:
                    energies = np.concatenate([energies, missing_energies], 0)

            if missing_forces is not None:
                if forces is None:
                    forces = missing_forces
                else:
                    forces = np.concatenate([forces, missing_forces], 0)

        return energies, forces


"""
TODO: 
- Implement get_energy_numpy, get_forces_numpy, and get_energy_and_forces_numpy()
- potentially allow framework-agnostic decorator to ignore self parameter to be directly applicable
"""


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
        "datasets/chrklitz99/alanine_dipeptide",
        length_unit="nanometer",
        n_workers=None,
    )
    target = MolecularBoltzmann.create_from_pdb(
        "test_bm", "target_cache/alanine_dipeptide/topology.pdb"
    )

    pos_min_energy = target.get_position_min_energy()
    print(pos_min_energy)

    print(target.get_log_prob(np.expand_dims(pos_min_energy, 0)))
