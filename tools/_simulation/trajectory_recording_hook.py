import gc
import json
import os

import numpy as np
from reform.simu_utils import (
    MultiTSimulation,
    OMMTReplicas,
    ReplicaExchangeHook,
    SimulationHook,
)

# from .numpy_append import NumpyAppendFile
from .h5_append import H5AppendFile


class H5RecorderHook(SimulationHook):
    """Recording the trajectory when called."""

    def __init__(
        self,
        npy_path: str,
        buffer_size: int = 1000,
        dtype=np.float32,
        save_traj_of_replicas: list[int] | None = None,
    ):
        assert (
            dtype is np.float32 or np.float64
        ), "Unrecognized dtype, should be either numpy.float32 or numpy.float64!"

        self.dtype = dtype
        self.npy_path = npy_path

        self.buffer = None
        self.buffer_size = buffer_size
        self.current_buffer_pos = 0

        self.curr_posis = None

        self.save_traj_of_replicas = save_traj_of_replicas
        if self.save_traj_of_replicas is not None:
            self.save_traj_of_replicas = sorted(self.save_traj_of_replicas)
            print(f"Only saving trajectories of replicas: {self.save_traj_of_replicas}")

        # Track total number of frames written to disk (flushed)
        self.total_frames_written = 0

    def action(self, context: OMMTReplicas) -> None:
        if self.curr_posis is None:
            self.curr_posis = context.get_all_positions_as_numpy()
            if self.save_traj_of_replicas is not None:
                self.curr_posis = self.curr_posis[self.save_traj_of_replicas, ...]
            self.curr_posis = self.curr_posis[np.newaxis, ...]
        else:
            if self.save_traj_of_replicas is not None:
                self.curr_posis[...] = context.get_all_positions_as_numpy()[
                    np.newaxis, self.save_traj_of_replicas, ...
                ]
            else:
                self.curr_posis[...] = context.get_all_positions_as_numpy()[
                    np.newaxis, ...
                ]

        if self.buffer is None:
            self.buffer = np.empty(
                shape=(self.buffer_size, *(self.curr_posis.shape[1:])),
                dtype=self.dtype,
            )

        self.buffer[self.current_buffer_pos : self.current_buffer_pos + 1, ...] = (
            self.curr_posis
        )
        self.current_buffer_pos += 1

        if self.current_buffer_pos == self.buffer_size:
            with H5AppendFile(self.npy_path) as file:
                file.append(self.buffer)
            self.total_frames_written += self.current_buffer_pos
            self.current_buffer_pos = 0

            gc.collect()

    def flush(self):
        if self.buffer is not None and self.current_buffer_pos > 0:
            with H5AppendFile(self.npy_path) as file:
                file.append(self.buffer[: self.current_buffer_pos])
            self.total_frames_written += self.current_buffer_pos
            self.current_buffer_pos = 0

    def get_total_frames(self) -> int:
        """Return total number of frames written to disk plus frames in buffer."""
        return self.total_frames_written + self.current_buffer_pos

    def __str__(self) -> str:
        return "Hook for storing trajectory to a npy file."


class CheckpointHook(SimulationHook):
    """Hook for saving checkpoints during simulation."""

    def __init__(
        self,
        simu: MultiTSimulation,
        output_dir: str,
        npy_hook: H5RecorderHook,
        time_step_fs: float,
    ):
        """
        Initialize the checkpoint hook.

        Args:
            simu: The MultiTSimulation object to save checkpoints for.
            output_dir: Directory where checkpoints will be saved.
            npy_hook: The NpyRecorderHook to flush when checkpointing.
            time_step_fs: Time step in femtoseconds.
        """
        self.simu = simu
        self.output_dir = output_dir
        self.npy_hook = npy_hook
        self.time_step_fs = time_step_fs
        self.chkpt_path = os.path.join(output_dir, "checkpoint.npz")
        self.metadata_path = os.path.join(output_dir, "checkpoint_metadata.json")

    def action(self, context: OMMTReplicas) -> None:
        # First, flush the trajectory buffer to disk
        self.npy_hook.flush()

        current_step = self.simu._current_step
        current_time_ps = current_step * self.time_step_fs / 1000.0
        total_frames = self.npy_hook.get_total_frames()

        # Save checkpoint (positions and velocities)
        self.simu.save_chkpt(self.chkpt_path)

        # Save metadata
        metadata = {
            "current_step": current_step,
            "current_time_ps": current_time_ps,
            "total_frames_written": total_frames,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"Checkpoint saved: step={current_step}, time={current_time_ps:.2f} ps, frames={total_frames}",
            flush=True,
        )

    def __str__(self) -> str:
        return "Hook for saving checkpoints."


def load_checkpoint_metadata(checkpoint_dir: str) -> dict:
    """Load checkpoint metadata from a directory."""
    metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Checkpoint metadata not found at {metadata_path}")
    with open(metadata_path, "r") as f:
        return json.load(f)


def truncate_trajectory(traj_path: str, num_frames: int) -> None:
    """
    Truncate a trajectory file to the specified number of frames.

    Args:
        traj_path: Path to the trajectory .npy file.
        num_frames: Number of frames to keep.
    """
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found at {traj_path}")

    traj = np.load(traj_path)
    # Trajectory shape: (n_frames, n_replicas, n_atoms, 3)
    if traj.shape[0] == num_frames:
        print(
            f"Trajectory already has exactly {num_frames} frames, no truncation needed."
        )
        return
    elif traj.shape[0] < num_frames:
        raise ValueError(
            f"Cannot truncate trajectory to {num_frames} frames, only has {traj.shape[0]} frames. This is likely an error."
        )

    print(
        f"Truncating copied resume trajectory from {traj.shape[0]} to {num_frames} frames."
    )
    truncated_traj = traj[:num_frames]
    np.save(traj_path, truncated_traj)


def recording_hook_setup(
    simu: MultiTSimulation,
    simu_time: float,
    recording_interval: float,
    output_path: str,
    exchange_interval: float = 0.0,
    save_traj_of_replicas: list[int] | None = None,
) -> tuple[int, H5RecorderHook]:
    """Calculate the recording and (optional) replica exchange intervals
    and register the correpsonding hooks in the given simulation object.

    Args:
        simu: MultiTSimulation object
        simu_time: intended simulation time for each replica in unit ps.
        recording_interval: recording interval in unit ps.
        output_path: path to the recording npy file.
        exchange_interval (optional): Interval in ps for replica exchange.
        save_traj_of_replicas (optional): If set, only save the trajectories of the specified replica indices (0-based). Otherwise, save all replicas.

    Returns:
        Tuple[int, NpyRecorderHook, ReplicaExchangeHook]: Number of simulation steps, the
        recording hook, and the replica exchange hook (None if exchange_interval <= 0).
    """

    TIME_STEP = simu.get_time_step()
    simu_steps = int(simu_time * 1_000 / TIME_STEP)
    recording_steps = int(recording_interval * 1_000 / TIME_STEP)
    record_hook = H5RecorderHook(
        output_path, buffer_size=1000, save_traj_of_replicas=save_traj_of_replicas
    )
    simu.register_regular_hook(record_hook, recording_steps)
    if exchange_interval > 0:
        exchange_steps = int(exchange_interval * 1_000 / TIME_STEP)
        re_hook = ReplicaExchangeHook()
        simu.register_regular_hook(re_hook, exchange_steps)
    else:
        re_hook = None
    return simu_steps, record_hook, re_hook
