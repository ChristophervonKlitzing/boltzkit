import argparse
import gc
import os
import pickle

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

from boltzkit.evaluation.sample_based.tica import visualize_tica, get_tica_hist
from boltzkit.utils.histogram import get_histogram_2d, visualize_histogram_2d
from boltzkit.utils.molecular.tica import create_deeptime_tica_model, get_tica_features

from boltzkit.targets.boltzmann import MolecularBoltzmann
from boltzkit.utils.pdf import save_pdf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute TICA projection(s) from a molecular dynamics trajectory."
    )

    parser.add_argument(
        "--traj_path",
        type=str,
        required=True,
        help="Path to a NumPy trajectory file (.npy) containing Cartesian coordinates. "
        "Expected shape: (n_frames, n_atoms, 3). Coordinates must be in nanometer (nm) "
        "Example: --traj_path ./data/trajectory.npy",
    )
    parser.add_argument(
        "--traj_total_sim_time_ns",
        type=float,
        required=True,
        help=(
            "Total physical simulation time represented by the trajectory in nanoseconds. "
            "This is used to determine the physical time between frames and convert "
            "lag times from picoseconds to frame indices.\n"
            "Example: if the trajectory contains 100000 frames from a 100 ns simulation:\n"
            "    --traj_total_sim_time_ns 100"
        ),
    )
    parser.add_argument(
        "--system_name",
        type=str,
        required=True,
        help=(
            "Name of the molecular system used to construct the topology via "
            "MolecularBoltzmann. The name must correspond to a system supported by "
            "boltzkit.\n"
            "Example: --system_name datasets/chrklitz99/alanine_dipeptide"
        ),
    )
    parser.add_argument(
        "--lag_time_ps",
        type=float,
        required=True,
        help=(
            "Lag time used for TICA in picoseconds. This determines the temporal "
            "separation between frames used to estimate time correlations.\n"
            "Example: --lag_time_ps 100"
        ),
    )
    parser.add_argument(
        "--dont_use_dihedrals",
        action="store_true",
        help=(
            "Disable dihedral angle features (phi, psi, omega) in the TICA input "
            "features. By default dihedral features are included."
        ),
    )
    parser.add_argument(
        "--dont_use_distances",
        action="store_true",
        help=(
            "Disable Cα-Cα distance features in the TICA input features. "
            "By default distance features are included."
        ),
    )
    parser.add_argument(
        "--dont_use_koopman",
        action="store_true",
        help=("Disable Koopman reweighting when estimating the TICA model."),
    )

    parser.add_argument(
        "--skipN",
        type=int,
        default=0,
        help=(
            "Number of initial frames to discard before analysis. "
            "This is typically used to remove equilibration or burn-in periods.\n"
            "Example: --skipN 1000"
        ),
    )
    parser.add_argument(
        "--subsample_factor",
        type=int,
        default=10,
        help=(
            "Subsample the trajectory by keeping every N-th frame. "
            "This reduces temporal correlations and decreases computational cost.\n"
            "Example: --subsample_factor 10 keeps frames 0,10,20,..."
        ),
    )

    parser.add_argument(
        "--timescale_plot",
        action="store_true",
        help=(
            "Instead of computing a single TICA projection, evaluate several lag "
            "times and generate an implied timescale plot. This helps selecting an "
            "appropriate lag time."
        ),
    )

    parser.add_argument(
        "--colormap",
        action="store_true",
        help=(
            "Show colored tica vectors together with the the 2D torsion marginal "
            "vectors (Ramachandran) using region dependent coloring"
        ),
    )

    args = parser.parse_args()
    return args


def map_tica_projections_to_color(tics: np.ndarray):
    from matplotlib.colors import hsv_to_rgb

    # maps 2D coordinates to colors
    x = tics[:, 0]
    y = tics[:, 1]
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()

    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)

    # Hue: changes with x
    # Saturation: full (1.0)
    # Value: changes with y
    h = x_norm  # hue cycles through 0-1
    s = np.ones(x.shape[0])  # full saturation
    v = 0.5 + 0.5 * y_norm  # brightness from 0.5 to 1.0

    hsv = np.stack([h, s, v], axis=1)  # shape (N, 3)
    rgb = hsv_to_rgb(hsv)
    return rgb


def plot_tica_and_ram_colored(
    color_arr: np.ndarray, tics: np.ndarray, rams: np.ndarray
):
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    axes[0, 0].scatter(tics[:, 0], tics[:, 1], c=color_arr, s=20, edgecolors="none")
    axes[0, 1].scatter(rams[:, 0], rams[:, 1], c=color_arr, s=20, edgecolors="none")

    tica_hist = get_histogram_2d(tics)
    visualize_histogram_2d(tica_hist, ax=axes[1, 0], cbar=False)

    ram_hist = get_histogram_2d(rams)
    visualize_histogram_2d(ram_hist, ax=axes[1, 1], cbar=False)

    plt.show()


def tica_model_creator_tool(args):
    ##### Prepare dataset #####

    xs: np.ndarray = np.load(args.traj_path)
    total_traj_len = xs.shape[0]
    print("Original dataset shape:", total_traj_len)

    xs = xs[args.skipN :, ...]
    print(f"Dataset shape after skipping {args.skipN} frames:", xs.shape)

    # Adjust the total simulation time to match the reduced trajectory length
    traj_reduced_sim_time_ns = (
        xs.shape[0] / total_traj_len
    ) * args.traj_total_sim_time_ns

    # Does not influence the physical simulation time of the trajectory
    xs = xs[:: args.subsample_factor]
    print(
        f"Dataset shape after subsampling by factor {args.subsample_factor}:",
        xs.shape,
    )

    # time per frame after subsampling in picoseconds
    ps_per_frame = traj_reduced_sim_time_ns / xs.shape[0] * 1000

    if args.timescale_plot == True:
        lag_times = [20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 5000]  # in ps
    else:
        lag_times = [args.lag_time_ps]

    traj_dir = os.path.dirname(args.traj_path)

    eigenvalues_tica_0 = []
    eigenvalues_tica_1 = []
    for lag_time_ps in lag_times:
        gc.collect()
        lag_n_frames = lag_time_ps / ps_per_frame
        lag_n_frames = int(round(lag_n_frames, 0))

        print(
            f"Lag time {lag_time_ps}ps in frames: ",
            lag_n_frames,
            "(rounded from ",
            lag_time_ps / ps_per_frame,
            ")",
        )

        bm = MolecularBoltzmann(args.system_name, n_workers=None)
        topology = bm.get_mdtraj_topology()

        trajectory = md.Trajectory(xyz=xs, topology=topology)
        features = get_tica_features(
            trajectory,
            use_dihedrals=not args.dont_use_dihedrals,
            use_distances=not args.dont_use_distances,
        )

        tica_model = create_deeptime_tica_model(
            trajectory,
            lagtime=lag_n_frames,
            dim=2,
            use_dihedrals=not args.dont_use_dihedrals,
            use_distances=not args.dont_use_distances,
            use_koopman=not args.dont_use_koopman,
        )
        tics = tica_model.transform(features)

        eigenvalues_tica_0.append(tica_model.singular_values[0])
        eigenvalues_tica_1.append(tica_model.singular_values[1])

        ##### Plot TICA free energy profile #####

        tica_hist = get_tica_hist(tics)
        tica_pdf = visualize_tica(tica_hist)

        if args.colormap:
            from boltzkit.evaluation.sample_based.torsion_marginals import (
                get_phi_psi_vectors,
            )

            color_arr = map_tica_projections_to_color(tics)
            phis, psis = get_phi_psi_vectors(xs, topology)

            for i in range(phis.shape[1]):
                # Only use backbone angle pair 0
                phis_i = phis[:, i]
                psis_i = psis[:, i]
                rams = np.stack([phis_i, psis_i], 1)

                plot_tica_and_ram_colored(color_arr, tics, rams)

        vis_path = os.path.join(
            traj_dir,
            f"tica_free_energy_lag_{lag_time_ps}_ps"
            + ("_no_dihedrals" if args.dont_use_dihedrals else "")
            + ("_no_distances" if args.dont_use_distances else "")
            + ("_no_koopman" if args.dont_use_koopman else "")
            + ".pdf",
        )
        save_pdf(tica_pdf, vis_path)

    if args.timescale_plot:
        eigenvalues_tica_0 = np.array(eigenvalues_tica_0)
        eigenvalues_tica_1 = np.array(eigenvalues_tica_1)

        implied_timescales_tica_0 = lag_times / (-np.log(eigenvalues_tica_0))
        implied_timescales_tica_1 = lag_times / (-np.log(eigenvalues_tica_1))

        plt.figure()
        plt.plot(lag_times, implied_timescales_tica_0, marker="o", label="TIC 1")
        plt.plot(lag_times, implied_timescales_tica_1, marker="o", label="TIC 2")
        plt.xlabel("Lag time (ps)")
        plt.ylabel("Implied Timescales (ps)")
        plt.legend()
        plt.savefig(
            os.path.join(
                traj_dir,
                f"tica_implied_timescales_vs_lagtime"
                + ("_no_dihedrals" if args.dont_use_dihedrals else "")
                + ("_no_distances" if args.dont_use_distances else "")
                + ("_no_koopman" if args.dont_use_koopman else "")
                + ".png",
            )
        )
        plt.close()

        exit()

    ##### Save the TICA model #####

    with open(
        os.path.join(
            traj_dir,
            f"tica_model_lag_{lag_time_ps}_ps"
            + ("_no_dihedrals" if args.dont_use_dihedrals else "")
            + ("_no_distances" if args.dont_use_distances else "")
            + ("_no_koopman" if args.dont_use_koopman else "")
            + ".pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(tica_model, f)


if __name__ == "__main__":
    args = parse_args()
    tica_model_creator_tool(args)
