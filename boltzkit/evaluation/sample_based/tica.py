"""
A part of this implementation is copied and adapted from
'https://github.com/aimat-lab/AnnealedBG/tree/adaptive_smoothing'.

----------------------------------------------------------------------------

MIT License

Copyright (c) 2025 Henrik Schopmans

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

----------------------------------------------------------------------------
"""

import numpy as np
import mdtraj as md
import deeptime as dt
from .wasserstein import compute_euclidean_wasserstein_1_2


SELECTION = "symbol == C or symbol == N or symbol == S"


def wrap(array: np.ndarray):
    return (np.sin(array), np.cos(array))


def distances(xyz: np.ndarray):
    distance_matrix_ca: np.ndarray = np.linalg.norm(
        xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1
    )
    n_ca = distance_matrix_ca.shape[-1]
    m, n = np.triu_indices(n_ca, k=1)
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca


def tica_features(
    trajectory: md.Trajectory,
    use_dihedrals: bool = True,
    use_distances: bool = True,
    selection: str = SELECTION,
):
    top: md.Topology = trajectory.topology
    trajectory = trajectory.atom_slice(top.select(selection))
    # n_atoms = trajectory.xyz.shape[1]
    if use_dihedrals:
        _, phi = md.compute_phi(trajectory)
        _, psi = md.compute_phi(trajectory)
        _, omega = md.compute_omega(trajectory)
        dihedrals = np.concatenate([*wrap(phi), *wrap(psi), *wrap(omega)], axis=-1)
    if use_distances:
        ca_distances = distances(trajectory.xyz)
    if use_distances and use_dihedrals:
        return np.concatenate([ca_distances, dihedrals], axis=-1)
    elif use_distances:
        return ca_distances
    else:
        return []


def process_tica_in_batches(
    data: np.ndarray,
    topology: md.Topology,
    tica_model: dt.decomposition.TransferOperatorModel,
) -> np.ndarray:
    data = data.reshape(data.shape[0], -1, 3)
    ticas_list = []
    for i in range(0, data.shape[0], 50000):
        tica_features_batch = tica_features(
            md.Trajectory(
                xyz=data[i : i + 50000],
                topology=topology,
            )
        )
        ticas_list.append(tica_model.transform(tica_features_batch))
    return np.concatenate(ticas_list, axis=0)


def compute_tica_projections(
    ground_truth_cart: np.ndarray,
    model_samples_cart: np.ndarray,
    topology: md.Topology,
    tica_model: dt.decomposition.TransferOperatorModel,
):
    ticas_ground_truth = process_tica_in_batches(
        ground_truth_cart,
        topology,
        tica_model,
    )
    ticas_model = process_tica_in_batches(
        model_samples_cart,
        topology,
        tica_model,
    )
    return ticas_ground_truth, ticas_model


def compute_tica_wasserstein_2(
    ground_truth_cart: np.ndarray,
    model_samples_cart: np.ndarray,
    topology: md.Topology,
    tica_model: dt.decomposition.TransferOperatorModel,
):
    tica_projections_gt, tica_projections_model = compute_tica_projections(
        ground_truth_cart, model_samples_cart, topology, tica_model
    )
    _, tica_W2 = compute_euclidean_wasserstein_1_2(
        tica_projections_model, tica_projections_gt, include_w1=False
    )
    return tica_W2


def plot_tica(
    ground_truth_cart: np.ndarray,
    model_samples_cart: np.ndarray,
    topology: md.Topology,
    tica_model: dt.decomposition.TransferOperatorModel,
):
    import matplotlib.pyplot as plt

    # TODO: This function is unfinished
    # It should return a pdf for plotting, as well as the raw histogram data.

    tica_projections_gt, tica_projections_model = compute_tica_projections(
        ground_truth_cart, model_samples_cart, topology, tica_model
    )

    # counts, bin_edges_x, bin_edges_y = np.histogram2d(
    #     tica_projections_gt[:, 0], tica_projections_gt[:, 1]
    # )
    _, (ax0, ax1) = plt.subplots(1, 2)
    ax0.hist2d(tica_projections_gt[:, 0], tica_projections_gt[:, 1], bins=100)
    ax1.hist2d(tica_projections_model[:, 0], tica_projections_model[:, 1], bins=100)
    # plt.hist(tica_projections_gt[:, 0], tica_projections_gt[:, 1], density=True)
    plt.show()


if __name__ == "__main__":
    import pickle

    # TODO: instead of loading a TICA model using pickle, its parameters should be loaded.
    # pickle can fail if the deeptime package changes substantially between versions.

    with open("tica.pkl", "rb") as f:
        tica_model: dt.decomposition.TransferOperatorModel = pickle.load(f)
    print(tica_model.get_params())
    from boltzkit.targets.boltzmann import MolecularBoltzmann

    bm = MolecularBoltzmann("datasets/chrklitz99/test_system")
    topology = bm.get_mdtraj_topology()

    model_samples = np.random.randn(1000, 66)
    gt_samples = np.random.randn(1000, 66)

    # print(tica_model, type(tica_model))
    plot_tica(
        gt_samples,
        model_samples,
        topology,
        tica_model,
    )
