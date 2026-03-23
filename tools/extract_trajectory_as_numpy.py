import argparse
import gc
import os
import numpy as np
import h5py
from pathlib import Path

import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible random permutation of trajectory data and save both the permuted data "
            "and the permutation.\n"
            "The input is a trajectory array `x` loaded from an HDF5 (.h5) or NumPy (.npy) file. "
            "A random permutation `perm` is generated from the specified seed (default: 0), along with its inverse.\n"
            "The tool applies the *inverse permutation* to the data:\n"
            "    x_permuted = x[inv_perm]\n"
            "Two output files are written:\n"
            "  - '*_permuted.npy': the permuted array `x_permuted`\n"
            "  - '*_perm.npy': the forward permutation `perm`\n"
            "The original data can be reconstructed via:\n"
            "    x == x_permuted[perm]\n"
            "By default, outputs are written to the input file's directory, but a custom output directory "
            "can be specified with --output."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to input HDF5 (.h5) or a npy file containing the trajectory dataset",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset key inside the HDF5 file (default: first dataset found). If input is a .npy file, this argument is ignored.",
    )

    parser.add_argument(
        "--skipN",
        type=int,
        default=0,
        help="Skip the first N samples in the trajectory and don't save them.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input file)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    parser.add_argument(
        "--trajectory_index",
        type=int,
        default=0,
        help="Selects the trajectory by index (default 0 is used)",
    )

    return parser.parse_args()


def load_dataset(file_path: Path, trajectory_idx: int, dataset_key: str | None = None):
    suffix = file_path.suffix.lower()
    if suffix == ".h5":
        with h5py.File(file_path, "r") as f:
            if dataset_key is None:
                keys = list(f.keys())
                if len(keys) == 1:
                    dataset_key = keys[0]
                else:
                    raise ValueError(
                        f"The file '{file_path}' contains multiple keys and no dataset_key is specified: {keys}"
                    )

            data = f[dataset_key][:]
    elif suffix == ".npy":
        data = np.load(file_path)

    if not len(data.shape) == 4:
        raise ValueError(
            "Input data must have shape (length, #trajectories, #atoms, 3)"
        )

    return data[:, trajectory_idx, :, :]


def main():
    args = parse_args()

    input_path = args.input
    output_dir = args.output or os.path.dirname(os.path.abspath(input_path))
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_dataset(input_path, args.trajectory_index, args.dataset)
    print(f"Loaded input trajectory of length {data.shape[0]}")

    print(f"Skip the first {args.skipN} samples...")
    data = data[args.skipN :]
    print(f"The remaining trajectory has length {data.shape[0]}")

    # Save outputs
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    traj_path = os.path.join(output_dir, f"{base_name}_traj{args.trajectory_index}.npy")

    print(f"Saving trajectory...")
    np.save(traj_path, data)
    print(f"Saved trajectory to: {traj_path}")


if __name__ == "__main__":
    main()
