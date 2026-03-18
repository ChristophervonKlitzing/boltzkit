#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import numpy as np
import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly subsample a trajectory without replacement into multiple splits.\n\n"
            "Given an input array x, this tool draws disjoint subsets (without replacement) "
            "according to user-specified sizes. Each subset is saved as a separate .npy file.\n\n"
            "Example: --splits 100 200 creates two subsets of size 100 and 200.\n"
            "The sum of all split sizes must be <= len(x)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to input HDF5 (.h5) or NumPy (.npy) file",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset key inside HDF5 file (ignored for .npy)",
    )

    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        required=True,
        help="List of split sizes (e.g. --splits 100 200 300)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same as input)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    return parser.parse_args()


def load_dataset(file_path: Path, dataset_key: str | None = None):
    suffix = file_path.suffix.lower()

    if suffix == ".h5":
        with h5py.File(file_path, "r") as f:
            if dataset_key is None:
                keys = list(f.keys())
                if len(keys) == 1:
                    dataset_key = keys[0]
                else:
                    raise ValueError(
                        f"Multiple datasets found. Specify --dataset. Keys: {keys}"
                    )
            data = f[dataset_key][:]

    elif suffix == ".npy":
        data = np.load(file_path)

    else:
        raise ValueError("Unsupported file type. Use .h5 or .npy")

    return data


def main():
    args = parse_args()

    input_path = args.input
    output_dir = args.output or Path(input_path).resolve().parent
    os.makedirs(output_dir, exist_ok=True)

    data = load_dataset(input_path, args.dataset)

    n = data.shape[0]
    splits = args.splits

    if sum(splits) > n:
        raise ValueError(f"Sum of splits ({sum(splits)}) exceeds dataset size ({n})")

    rng = np.random.default_rng(args.seed)

    # Draw all indices without replacement
    indices = rng.permutation(n)

    base_name = input_path.stem

    start = 0
    for i, size in enumerate(splits):
        end = start + size
        subset_idx = indices[start:end]
        subset = data[subset_idx]

        out_path = output_dir / f"{base_name}_split{i}_n{size}.npy"
        np.save(out_path, subset)

        print(f"Saved split {i} (n={size}) to: {out_path}")

        start = end


if __name__ == "__main__":
    main()
