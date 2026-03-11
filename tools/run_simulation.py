import argparse
import os
import shutil

import time
from typing import Literal
import uuid
from datetime import datetime

import numpy as np
from reform import simu_utils

import sys

# Allows implementation packages under tools/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tools._simulation.observables_hook import ObservablesHook
from tools._simulation.status_hook import StatusHook
from tools._simulation.trajectory_recording_hook import (
    CheckpointHook,
    load_checkpoint_metadata,
    recording_hook_setup,
    truncate_trajectory,
)
import openmm as mm

from boltzkit.targets.boltzmann import MolecularBoltzmann


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a replica exchange molecular dynamics simulation."
    )
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        help="The name of the system to simulate, e.g., datasets/chrklitz99/test_system",
    )
    parser.add_argument(
        "--temps",
        type=str,
        default=None,
        help="Temperature specification. If not set, uses default np.geomspace(300.0,500.0,6). "
        "If a float, uses single temperature with no exchange. "
        "If format <min_temp>:<max_temp>:<NO_replicas>, creates geometric spacing.",
    )
    parser.add_argument(
        "--time_step",
        type=float,
        default=2.0,
        help="The time step in fs.",
    )
    parser.add_argument(
        "--rec_interval",
        type=float,
        default=0.1,
        help="The recording interval in ps.",
    )
    parser.add_argument(
        "--exchange_interval",
        type=float,
        default=1.0,
        help="The exchange interval in ps.",
    )
    parser.add_argument(
        "--pre_eq_time",
        type=float,
        default=200.0,
        help="The pre-equilibration time in ns.",
    )
    parser.add_argument(
        "--simu_time",
        type=float,
        default=1200.0,
        help="The simulation time in ns.",
    )
    parser.add_argument(
        "--integrator",
        type=str,
        default="Langevin",
        choices=["Langevin", "LangevinMiddle"],
        help="The integrator to use. Options: Langevin, LangevinMiddle",
    )
    parser.add_argument(
        "--save_traj_of_replicas",
        nargs="*",
        type=int,
        default=None,
        help="If set, only save the trajectories of the specified replica indices (0-based). Otherwise, save all replicas.",
    )
    parser.add_argument(
        "--starting_configuration",
        type=str,
        default=None,
        help="Path to a npy file containing the starting configuration. Starting configuration should be given in nm.",
    )
    parser.add_argument(
        "--reset_to_starting_configuration_every",
        type=float,
        default=None,
        help="Reset configuration to starting configuration every X ns.",
    )
    parser.add_argument(
        "--use_scratch",
        action="store_true",
        help="Write intermediate output into $SCRATCH before copying it back to ./output/ at the end of the simulation.",
    )
    parser.add_argument(
        "--write_checkpoint_every_ns",
        type=float,
        default=None,
        help="If set, write a checkpoint every X ns during the main simulation (not during equilibration)."
        "Checkpoints include positions, velocities, and metadata about the current simulation state.",
    )
    parser.add_argument(
        "--resume_from_dir",
        type=str,
        default=None,
        help="Path to an output directory of a previous (crashed) simulation to resume from."
        "A new output directory will be created, and the checkpoint will be loaded and continued.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="CUDA",
        help="Platform to run MD on. Another option is 'CPU'.",
    )
    args = parser.parse_args()

    if args.starting_configuration is not None and args.pre_eq_time != 0.0:
        raise ValueError(
            "If starting_configuration is provided, pre_eq_time must be 0."
        )

    if args.write_checkpoint_every_ns is not None:
        checkpoint_interval_ps = args.write_checkpoint_every_ns * 1000  # ns to ps
        if int(checkpoint_interval_ps * 1000) % int(args.rec_interval * 1000) != 0:
            raise ValueError(
                f"Checkpoint interval ({args.write_checkpoint_every_ns} ns = {checkpoint_interval_ps} ps) "
                f"must be a multiple of recording interval ({args.rec_interval} ps)."
            )

    return args


def run_remd(
    args,
    system: mm.System,
    initial_position: list[mm.Vec3] | np.ndarray,
    interface: Literal["single_threaded", "replicated_system"] = "single_threaded",
    platform: str = "CUDA",
):
    TIME_STEP = args.time_step  # in fs
    PRE_EQ_TIME = args.pre_eq_time * 1000  # convert from ns to ps
    SIMU_TIME = args.simu_time * 1000  # convert from ns to ps
    RECORDING_INTERVAL = args.rec_interval  # in ps
    EXCHANGE_INTERVAL = args.exchange_interval  # in ps

    # Set up temperatures based on the temps argument
    if args.temps is None:
        # Default case: use geometric spacing from 300.0 to 500.0 with 6 replicas
        TEMPS = np.geomspace(300.0, 500.0, 6)
    elif ":" in args.temps:
        # Format: <min_temp>:<max_temp>:<NO_replicas>
        parts = args.temps.split(":")
        if len(parts) != 3:
            raise ValueError(
                "temps format with colons must be <min_temp>:<max_temp>:<NO_replicas>"
            )
        min_temp = float(parts[0])
        max_temp = float(parts[1])
        n_replicas = int(parts[2])
        TEMPS = np.geomspace(min_temp, max_temp, n_replicas)
    else:
        # Single temperature case
        try:
            single_temp = float(args.temps)
            TEMPS = [single_temp]
            EXCHANGE_INTERVAL = 0
        except ValueError:
            raise ValueError(
                f"Invalid temps value: {args.temps}. Must be a float or <min_temp>:<max_temp>:<NO_replicas> format."
            )
    print(f"Running REMD with replicas at temperatures: {repr(TEMPS)}")

    # Validate integrator choice
    if args.integrator == "LangevinMiddle" and interface != "single_threaded":
        raise ValueError(
            "LangevinMiddle integrator can currently only be used with the single_threaded interface."
        )

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    random_id = uuid.uuid4().hex[:8]

    final_output_dir = os.path.join("./output", args.system, f"{timestamp}_{random_id}")
    if args.use_scratch:
        scratch_root = os.environ.get("SCRATCH")
        if not scratch_root:
            raise EnvironmentError(
                "SCRATCH environment variable must be set when --use_scratch is used."
            )
        OUTPUT_DIR = os.path.join(scratch_root, args.system, f"{timestamp}_{random_id}")
    else:
        OUTPUT_DIR = final_output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATH = f"{OUTPUT_DIR}/traj.npy"

    # Resuming from a previous simulation
    resume_metadata = None
    if args.resume_from_dir is not None:
        print(f"Resuming from previous simulation: {args.resume_from_dir}", flush=True)

        # Load checkpoint metadata
        resume_metadata = load_checkpoint_metadata(args.resume_from_dir)
        print(f"Checkpoint metadata: {resume_metadata}", flush=True)

        # Copy trajectory file from old directory to the new one
        old_traj_path = os.path.join(args.resume_from_dir, "traj.npy")
        if os.path.exists(old_traj_path):
            shutil.copy2(old_traj_path, OUTPUT_PATH)
            # Truncate trajectory to the number of frames at checkpoint time:
            truncate_trajectory(OUTPUT_PATH, resume_metadata["total_frames_written"])
        else:
            raise FileNotFoundError(f"Trajectory file not found at {old_traj_path}")

        # Copy observables.csv with a renamed file
        old_observables_path = os.path.join(args.resume_from_dir, "observables.csv")
        if os.path.exists(old_observables_path):
            new_observables_path = os.path.join(
                OUTPUT_DIR, "observables_resumed_from.csv"
            )
            shutil.copy2(old_observables_path, new_observables_path)
            print(f"Copied observables to {new_observables_path}")
        else:
            raise FileNotFoundError(
                f"Observables file not found at {old_observables_path}"
            )

    n_replicas = len(TEMPS)

    integrator_params = {
        "integrator": args.integrator,
        "friction_in_inv_ps": 1.0,
        "time_step_in_fs": TIME_STEP,
    }

    simu = simu_utils.MultiTSimulation(
        system,
        TEMPS,
        interface=interface,
        integrator_params=integrator_params,
        verbose=False,
        platform=platform,
    )

    print(
        f"Performing RE-MD simulation of {args.system} system\n",
        f"Temperatures: {repr(TEMPS)}\n",
        f"Time step: {repr(TIME_STEP)} fs\n",
        f"Integrator: {args.integrator}\n",
        f"Simulation time: {repr(SIMU_TIME)} ps ({args.simu_time} ns)\n",
        f"Pre-equilibration time: {repr(PRE_EQ_TIME)} ps ({args.pre_eq_time} ns)\n",
        f"Exchange interval: {repr(EXCHANGE_INTERVAL)} ps\n",
        f"Output path: {OUTPUT_PATH}\n",
        f"Platform: {platform}\n",
        flush=True,
    )

    simu.set_positions([initial_position] * n_replicas)

    if resume_metadata is not None:
        print(
            "Resuming from checkpoint, skipping energy minimization and equilibration...",
            flush=True,
        )
        # Load checkpoint (positions and velocities)
        chkpt_path = os.path.join(args.resume_from_dir, "checkpoint.npz")
        simu.load_chkpt(chkpt_path, check_temps=True)
        print(f"Loaded checkpoint from {chkpt_path}", flush=True)

        eq_steps = 0  # No equilibration when resuming
        _simu_steps = int(SIMU_TIME * 1000 / TIME_STEP)

        # Calculate remaining steps from where we left off
        resumed_step = resume_metadata["current_step"]
        remaining_simu_steps = _simu_steps - resumed_step
        print(
            f"Resuming from step {resumed_step}, running {remaining_simu_steps} more steps",
            flush=True,
        )
    else:
        print("Performing energy minimization...", flush=True)
        start = time.time()
        simu.minimize_energy()
        print(f"Done. Elapsed time: {time.time() - start} s.", flush=True)

        simu.set_velocities_to_temp()

        eq_steps = int(PRE_EQ_TIME * 1000 / TIME_STEP)
        _simu_steps = int(SIMU_TIME * 1000 / TIME_STEP)
        remaining_simu_steps = _simu_steps

    status_hook = StatusHook(total_steps=eq_steps + _simu_steps)
    simu.register_regular_hook(status_hook, 50000)

    observable_hook = ObservablesHook(csv_path=f"{OUTPUT_DIR}/observables.csv")
    simu.register_regular_hook(observable_hook, 100000)

    status_hook.start_timer()
    if eq_steps > 0:
        print("Performing pre-equilibration...", flush=True)
        start = time.time()
        simu.run(eq_steps)  # Pre-equilibration
        print(f"Done. Elapsed time: {time.time() - start} s.", flush=True)

    # When resuming, restore the step counter to where we left off.
    # This ensures checkpoints save the correct absolute step number.
    if resume_metadata is not None:
        simu._current_step = resume_metadata["current_step"]
    else:
        simu.reset_step_counter()

    # Install the simulation hooks for recording and exchange:
    simu_steps, npy_hook, re_hook = recording_hook_setup(
        simu=simu,
        simu_time=SIMU_TIME,
        recording_interval=RECORDING_INTERVAL,
        output_path=OUTPUT_PATH,
        exchange_interval=EXCHANGE_INTERVAL,
        save_traj_of_replicas=args.save_traj_of_replicas,
    )

    observable_hook.set_re_hook(re_hook)
    assert simu_steps == _simu_steps

    # Register checkpoint hook if requested
    if args.write_checkpoint_every_ns is not None:
        checkpoint_time_ps = args.write_checkpoint_every_ns * 1000
        checkpoint_steps = int(checkpoint_time_ps * 1000 / TIME_STEP)
        checkpoint_hook = CheckpointHook(
            simu=simu,
            output_dir=OUTPUT_DIR,
            npy_hook=npy_hook,
            time_step_fs=TIME_STEP,
        )
        simu.register_regular_hook(checkpoint_hook, checkpoint_steps)
        print(
            f"Checkpointing every {args.write_checkpoint_every_ns} ns ({checkpoint_steps} steps)",
            flush=True,
        )

    # When resuming, update the npy_hook to track the already written frames:
    if resume_metadata is not None:
        npy_hook.total_frames_written = resume_metadata["total_frames_written"]
        print(
            f"Resuming trajectory with {npy_hook.total_frames_written} frames already written."
        )

    if args.starting_configuration is not None:
        print(
            f"Loading starting configuration from {args.starting_configuration}...",
        )
        loaded_pos = np.load(args.starting_configuration).reshape(-1, 3)
        starting_positions = [loaded_pos] * n_replicas

        simu.set_positions(starting_positions)
        simu.set_velocities_to_temp()

        if args.reset_to_starting_configuration_every is not None:
            reset_time_ns = float(args.reset_to_starting_configuration_every)
            reset_time_ps = reset_time_ns * 1000
            reset_steps = int(reset_time_ps * 1000 / TIME_STEP)
            print(
                f"Registering reset hook every {reset_time_ns} ns (Every {reset_steps} steps, total number of resets: {simu_steps // reset_steps})",
                flush=True,
            )

            class ResetConfigurationHook(simu_utils.SimulationHook):
                def __init__(self, positions):
                    super().__init__()
                    self.positions = positions

                def action(self, context) -> None:
                    simu.set_positions(self.positions)
                    simu.set_velocities_to_temp()

                def __str__(self) -> str:
                    return (
                        "Hook for resetting configurations to starting configuration."
                    )

            reset_hook = ResetConfigurationHook(starting_positions)
            simu.register_regular_hook(reset_hook, reset_steps)

    print("Starting simulation...", flush=True)
    start = time.time()
    simu.run(remaining_simu_steps)
    print(f"Done. Elapsed time: {time.time() - start} s.", flush=True)

    npy_hook.flush()

    if OUTPUT_DIR != final_output_dir:
        print(
            f"Copying simulation data from {OUTPUT_DIR} to {final_output_dir}...",
            flush=True,
        )
        os.makedirs(os.path.dirname(final_output_dir), exist_ok=True)
        shutil.copytree(OUTPUT_DIR, final_output_dir, dirs_exist_ok=True)


if __name__ == "__main__":
    args = parse_args()
    system = MolecularBoltzmann(args.system, n_workers=None)
    args.system = args.system.split("/")[-1]

    run_remd(
        args,
        system.get_openmm_system(),
        system.get_position_min_energy().reshape(-1, 3),
        platform=args.platform,
    )
