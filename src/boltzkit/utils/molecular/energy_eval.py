from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
import math
import os
import signal
import sys
from typing import Literal
import numpy as np
import openmm as mm
from openmm import unit, app
import atexit
from boltzkit.utils.molecular.conversion import vec3_list_to_numpy

import multiprocessing as mp


# Use of electronvolt instead of joule to decrease the effect of rounding errors.
# The resulting units cancel out when computing the Boltzmann density (dividing by kB[eV/T] * temperature[K]).
kB_in_eV_per_K = 8.617333262145177e-05  # Boltzmann constant in eV/K
kJ_per_mol_to_eV = 0.010364269656262174  # converts KJ/mol -> eV

# repeated creation of new units through binary operators
# can potentially result in indefinite memory growth.
# -> define force unit outside of `evaluate_energy_single`, which will be called many times.
force_unit = unit.kilojoule_per_mole / unit.nanometer


@contextmanager
def _silence_stderr():
    old = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old


def _test_ABI_set_positions_compatibility(sim: app.Simulation):
    """
    Test whether positions can be safely set in an OpenMM Simulation when using NumPy >= 2.0.0.

    OpenMM may have been compiled against NumPy < 2.0.0 while the current environment
    uses NumPy >= 2.0.0. This can lead to a C-ABI mismatch, which may trigger warnings
    or errors if NumPy arrays are passed between Python and OpenMM (e.g., using
    `asNumpy=True` or passing `ndarray` inputs).

    This function performs a test by attempting to set positions using the simulation's
    context and verifying that the positions are correctly applied.

    Parameters
    ----------
    sim : app.Simulation
        The OpenMM Simulation instance to test for safe position setting.

    Raises
    ------
    RuntimeError
        If the test fails, indicating that positions cannot be safely set with the
        current OpenMM/NumPy combination. This may happen due to a NumPy C-ABI
        mismatch (OpenMM compiled with NumPy < 2.0.0 while NumPy >= 2.0.0 is installed).
        In that case, consider:
            - downgrading NumPy to < 2.0, or
            - rebuilding OpenMM against NumPy >= 2.0.
    """

    s: mm.State = sim.context.getState(getPositions=True)
    p_old = s.getPositions(asNumpy=True)

    test_offset = np.arange(p_old.size).reshape(p_old.shape)
    p_test = p_old.copy() + test_offset

    def set_positions_impl(sim: app.Simulation, pos: np.ndarray):
        sim.context.setPositions(pos)

    with _silence_stderr():
        set_positions_impl(sim, p_test)

    s: mm.State = sim.context.getState(getPositions=True)
    p_probe = s.getPositions(asNumpy=True)

    if np.allclose(p_probe, p_test):
        set_positions_impl(sim, p_old)
    else:
        raise RuntimeError(
            "ABI test failed: positions could not be safely set in this OpenMM simulation. "
            "This may be caused by a NumPy C-ABI mismatch (OpenMM compiled with NumPy < 2.0.0 "
            "while NumPy >= 2.0.0 is installed). Consider:\n"
            "  - downgrading NumPy to < 2.0, or\n"
            "  - rebuilding OpenMM against NumPy >= 2.0."
        )


def evaluate_energy_single(
    sim: app.Simulation,
    pos: list[mm.Vec3] | np.ndarray | unit.Quantity,
    include_energy: bool = True,
    include_forces: bool = True,
) -> np.ndarray:
    """
    Evaluate energy for a set of atomic positions specified in nanometers (nm).

    :param sim: Description
    :type sim: app.Simulation
    :param pos: Atomic positions as list of vectors or numpy array of shape (n_atoms, 3)
    :type pos: list[mm.Vec3] | np.ndarray | unit.Quantity
    :param include_energy: Whether to compute energy
    :type include_energy: bool
    :param include_forces: Whether to compute forces
    :type include_forces: bool
    :return: Return optional energy (in eV) and forces in (eV/nm)
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """

    sim.context.setPositions(pos)
    state: mm.State = sim.context.getState(
        getEnergy=include_energy, getForces=include_forces
    )

    if include_energy:
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        energy = energy * kJ_per_mol_to_eV
    else:
        energy = None

    if include_forces:
        forces = state.getForces().value_in_unit(force_unit)
        forces = vec3_list_to_numpy(forces)
        forces = forces * kJ_per_mol_to_eV
    else:
        forces = None

    return energy, forces


def get_openmm_platform(platform_name: Literal["CPU", "CUDA"] | None = None):
    def _create_cuda_platform():
        platform: mm.Platform = mm.Platform.getPlatformByName("CUDA")
        platform.setPropertyDefaultValue("DeviceIndex", "0")
        platform.setPropertyDefaultValue("Precision", "mixed")
        return platform

    if platform_name is None:
        try:
            return _create_cuda_platform()
        except mm.OpenMMException:
            platform_name = "CPU"

    if platform_name == "CPU":
        platform = mm.Platform.getPlatformByName("CPU")
    elif platform_name == "CUDA":
        platform = _create_cuda_platform()
    else:
        raise ValueError(f"Unknown platform name '{platform_name}'")

    return platform


def create_simulation(
    topology: app.Topology,
    system: mm.System,
    platform_name: Literal["CPU", "CUDA"] | None = None,
    integrator: mm.Integrator | None = None,
):
    """
    Create a simulation object using the given parameters.

    :param topology: Topology
    :type topology: openmm.app.Topology
    :param system: System
    :type system: openmm.System
    :param platform_name: Executation platform. None means the platform is determined automatically, using CUDA if available.
    :type platform_name: Literal["CPU", "CUDA"] | None
    :param integrator: Optional integrator (default is None). This is only relevant if the resulting simulation object is used
    for something else than static energy evaluation. In this case, a Langevin integrator is used a dummy object.
    :type integrator: mm.Integrator | None
    """
    if integrator is None:
        integrator = mm.LangevinIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.femtosecond
        )  # integrator just to create simulation object ( -> temperature irrelevant)

    platform = get_openmm_platform(platform_name)

    sim = app.Simulation(
        topology=topology,
        system=system,
        integrator=integrator,
        platform=platform,
    )
    return sim


class EnergyEval(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def evaluate_batch(
        self, x: np.ndarray, include_energy: bool = True, include_forces: bool = True
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        raise NotImplementedError


class SequentialEnergyEval(EnergyEval):
    def __init__(
        self,
        topology: app.Topology,
        system: mm.System,
        platform: Literal["CPU", "CUDA"] | None = "CPU",
    ):
        super().__init__()

        self._topology = topology
        self._system = system

        self._simulation = create_simulation(
            topology=topology, system=system, platform_name=platform
        )
        _test_ABI_set_positions_compatibility(self._simulation)

    def _evaluate_batch_simple(
        self, x: np.ndarray, include_energy: bool, include_forces: bool
    ):
        if include_energy:
            batch_energies = np.empty(x.shape[0])
        else:
            batch_energies = None

        if include_forces:
            batch_forces = np.empty_like(x)
        else:
            batch_forces = None

        for i in range(x.shape[0]):
            energy, forces = evaluate_energy_single(
                self._simulation,
                x[i],
                include_energy=include_energy,
                include_forces=include_forces,
            )

            if include_energy:
                batch_energies[i] = energy

            if include_forces:
                batch_forces[i] = forces

        return batch_energies, batch_forces

    def evaluate_batch(self, x, include_energy=True, include_forces=True):
        assert (
            include_energy or include_forces
        ), "Computing neither energy nor forces makes no sense"

        input_shape = x.shape
        batch = input_shape[0]

        if len(input_shape) == 2:
            x = x.reshape((batch, -1, 3))

        energies, forces = self._evaluate_batch_simple(
            x, include_energy=include_energy, include_forces=include_forces
        )
        if include_forces:
            forces = forces.reshape(input_shape)
        return energies, forces


# This gets initialized per worker process
_worker_energy_eval: SequentialEnergyEval | None = None


def _init_worker(topology: app.Topology, system: mm.System, platform):
    # Ignore SIGINTs (e.g., Ctrl+C) to currectly shutdown worker pool
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    global _worker_energy_eval
    _worker_energy_eval = SequentialEnergyEval(
        topology=topology, system=system, platform=platform
    )


def _eval_worker(x: np.ndarray, include_energy: bool, include_forces: bool):
    return _worker_energy_eval.evaluate_batch(
        x, include_energy=include_energy, include_forces=include_forces
    )


class ParallelEnergyEval:
    def __init__(
        self,
        topology: app.Topology,
        system: mm.System,
        platform: Literal["CPU", "CUDA"] | None = "CPU",
        n_workers: int = -1,
    ):
        super().__init__()

        if n_workers == -1:
            n_workers = mp.cpu_count()

        self.n_workers = n_workers
        print(f"Create parallel energy evaluation with {self.n_workers} workers")

        # fork will likely lead to issues with cuda -> use spawn
        cxt = mp.get_context("spawn")

        self.pool = cxt.Pool(
            processes=self.n_workers,
            initializer=_init_worker,
            initargs=(topology, system, platform),
        )

        atexit.register(self.pool.close)

    def evaluate_batch(
        self, x: np.ndarray, include_energy=True, include_forces=True, fragmentation=1
    ):
        """
        fragmentation (int, optional, default=1): Controls how samples are divided across workers.

        By default, N samples are evenly split among K workers (blocks of size N/K), which minimizes
        interprocess communication when all workers have similar speeds.

        If some workers are slower, they can become a bottleneck. Setting `fragmentation` > 1 splits
        the N samples into smaller blocks of size N / (K * fragmentation). This increases communication
        overhead but allows faster workers to process more blocks than slower ones, helping balance
        the workload.
        """

        assert (
            include_energy or include_forces
        ), "Computing neither energy nor forces makes no sense"

        n_total = x.shape[0]
        n_per_worker = n_total // self.n_workers
        if n_total % self.n_workers > 0:
            n_per_worker += 1  # distribute remaining
        # n_full_workers = n_total // n_per_worker
        split_index = list(
            range(n_per_worker, n_total, max(n_per_worker // fragmentation, 1))
        )
        x_split = np.split(x, split_index, 0)

        f = partial(
            _eval_worker, include_energy=include_energy, include_forces=include_forces
        )
        try:
            results = self.pool.map(f, x_split)
        except KeyboardInterrupt:
            print("Caught Ctrl+C, terminating process pool...")
            self.pool.terminate()
            self.pool.join()
            print("terminated")
            raise  # re-raise exception

        energies = [r[0] for r in results]
        forces = [r[1] for r in results]

        if include_energy:
            energies: np.ndarray = np.concatenate(energies, 0)
        else:
            energies = None

        if include_forces:
            forces: np.ndarray = np.concatenate(forces, 0)
        else:
            forces = None

        return energies, forces


if __name__ == "__main__":
    pdb_file_path = "notebooks/data/alanine-dipeptide-implicit.pdb"
    pdb = app.PDBFile(pdb_file_path)
    forcefield = app.ForceField("amber14-all.xml")
    system: mm.System = forcefield.createSystem(pdb.topology)

    energy_eval = ParallelEnergyEval(pdb.topology, system)

    np_positions = vec3_list_to_numpy(pdb.positions)

    energy, forces = energy_eval.evaluate_batch(np.stack([np_positions] * 4, 0))
    print("energies:", energy / (kB_in_eV_per_K * 300), type(energy))
    print("forces:", forces.shape if forces is not None else None, type(forces))

    # TODO: Compare with bgmol implementation and CV diffusion repo implementation
    # TODO: Test speed with different batch sizes on more threads

    check_force_scale = False
    if check_force_scale:
        # numerically check if force is scaled correctly
        shift = 0.000001 * np.random.randn(22, 3)
        pos_shifted = np_positions + shift
        energy, forces = energy_eval.evaluate_batch(np.expand_dims(np_positions, 0))
        energy_shifted, _ = energy_eval.evaluate_batch(np.expand_dims(pos_shifted, 0))
        v1 = energy_shifted - energy
        v2 = -np.dot(forces.flatten(), shift.flatten())
        print("Result should be 1:", v1 / v2)

    measure_time = False
    if measure_time:
        import timeit

        np.infty = np.inf
        from bgflow.distribution.energy.openmm import MultiContext

        n_times = 5
        batch = 16368

        seq_energy_eval = SequentialEnergyEval(pdb.topology, system)
        seq_time = timeit.timeit(
            lambda: seq_energy_eval.evaluate_batch(np.random.randn(batch, 22, 3)),
            number=n_times,
        )

        # Evaluation of 10^7 samples
        # ≈ 60min for sequential energy eval
        # ≈ 16min for parallel energy eval (with 6 worker threads on my laptop)

        integrator = mm.LangevinIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.femtosecond
        )
        c = MultiContext(energy_eval.n_workers, system, integrator, "CPU")
        c.evaluate(np.random.randn(batch, 22, 3))

        # with _silence_stderr():
        bgflow_time = timeit.timeit(
            lambda: c.evaluate(np.random.randn(batch, 22, 3)),
            number=n_times,
        )

        parallel_time = timeit.timeit(
            lambda: energy_eval.evaluate_batch(
                np.random.randn(batch, 22, 3), fragmentation=batch
            ),
            number=n_times,
        )

        print(f"parallel: {parallel_time / n_times:.3f}s for {batch} samples")
        print(f"sequential: {seq_time / n_times:.3f}s for {batch} samples")
        print(f"bgflow: {bgflow_time / n_times:.3f}s for {batch} samples")
        # print(f"parallel speedup: {seq_time / parallel_time:.2f}x")
        # print(f"bgflow speedup: {seq_time / bgflow_time:.2f}x")
        c.terminate()
