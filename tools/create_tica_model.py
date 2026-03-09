import argparse
import os
import pickle

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

from boltzkit.evaluation.sample_based.tica import visualize_tica, get_tica_hist
from boltzkit.utils.molecular.tica import create_tica_model, get_tica_features

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
            "Example: --system_name datasets/chrklitz99/test_system"
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
        "--skip_N",
        type=int,
        default=0,
        help=(
            "Number of initial frames to discard before analysis. "
            "This is typically used to remove equilibration or burn-in periods.\n"
            "Example: --skip_N 1000"
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

    args = parser.parse_args()
    return args


def tica_model_creator_tool(args):
    ##### Prepare dataset #####

    xs: np.ndarray = np.load(args.traj_path)
    total_traj_len = xs.shape[0]
    print("Original dataset shape:", total_traj_len)

    xs = xs[args.skip_N :, ...]
    print(f"Dataset shape after skipping {args.skip_N} frames:", xs.shape)

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
        lag_times = [20, 50, 100, 200, 500, 1000, 2000, 5000]  # in ps
    else:
        lag_times = [args.lag_time_ps]

    traj_dir = os.path.dirname(args.traj_path)

    eigenvalues_tica_0 = []
    eigenvalues_tica_1 = []
    for lag_time_ps in lag_times:
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

        tica_model = create_tica_model(
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
