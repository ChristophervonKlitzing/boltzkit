from typing import Literal, TypeAlias
import numpy as np

from boltzkit.utils.molecular.marginals import get_phi_psi_vectors
import mdtraj as md
from boltzkit.evaluation.sample_based.histogram import (
    Histogram1D,
    Histogram2D,
    get_histogram_1d,
    get_histogram_2d,
)
from .wasserstein import get_torus_wasserstein as _get_torus_wasserstein

T: TypeAlias = dict[Literal["phi_psi", "phi", "psi"], Histogram1D | Histogram2D]


def get_torsion_angles(samples: np.ndarray, topology: md.Topology):
    # An alias (here for completeness)
    return get_phi_psi_vectors(samples, topology)


def get_torsion_marginal_hists(phis: np.ndarray, psis: np.ndarray, **kwargs) -> list[T]:
    angles = np.stack([phis, psis], -1)  # (batch, <num-angle-pairs>, 2)

    range_1d = (-np.pi, np.pi)
    range_2d = ((-np.pi, np.pi), (-np.pi, np.pi))

    torsion_marginals: list[T] = []
    for angle_pair_idx in range(angles.shape[1]):
        marginals: T = {}
        # 2D histogram
        hist_2d = get_histogram_2d(
            angles[:, angle_pair_idx, :], data_range=range_2d, **kwargs
        )
        marginals["phi_psi"] = hist_2d

        # 2 1D histograms
        phi_hist_1d = get_histogram_1d(
            angles[:, angle_pair_idx, 0], data_range=range_1d, **kwargs
        )
        marginals["phi"] = phi_hist_1d

        psi_hist_1d = get_histogram_1d(
            angles[:, angle_pair_idx, 1], data_range=range_1d, **kwargs
        )
        marginals["psi"] = psi_hist_1d

        torsion_marginals.append(marginals)

    return torsion_marginals


# Old but maybe useful for documentation
def compute_torus_wasserstein_2(
    ground_truth_cart: np.ndarray,
    model_samples_cart: np.ndarray,
    topology: md.Topology,
) -> float:
    """
    Compute the toroidal 2-Wasserstein distance between backbone torsion
    angle distributions of two molecular ensembles.

    Parameters
    ----------
    ground_truth_cart : np.ndarray
        Cartesian coordinates of the reference ensemble with shape
        (batch, n_atoms, 3) or (batch, n_atoms * 3)
    model_samples_cart : np.ndarray
        Cartesian coordinates of the model-generated ensemble with shape
        (batch, n_atoms, 3) or (batch, n_atoms * 3)
    topology : md.Topology
        Molecular topology corresponding to the atomic coordinates.

    Returns
    -------
    float
        The toroidal 2-Wasserstein distance between the joint φ/ψ
        angle distributions of the two ensembles.

    Notes
    -----
    - Backbone φ and ψ torsion angles are extracted for each element in the batch.
    - The joint angular distribution is treated as living on a torus.
    - Uniform weights are assumed for both ensembles.
    """

    if len(ground_truth_cart.shape) == 2:
        batch = ground_truth_cart.shape[0]
        ground_truth_cart = ground_truth_cart.reshape((batch, -1, 3))
        model_samples_cart = model_samples_cart.reshape((batch, -1, 3))

    phis_true, psis_true = get_phi_psi_vectors(ground_truth_cart, topology)
    angles_true = np.concatenate([phis_true, psis_true], axis=1)

    phis_pred, psis_pred = get_phi_psi_vectors(model_samples_cart, topology)
    angles_pred = np.concatenate([phis_pred, psis_pred], axis=1)

    t_wasserstein_2 = _get_torus_wasserstein(angles_true, angles_pred)
    return t_wasserstein_2


def get_torus_wasserstein_2(
    phis_psis_true: tuple[np.ndarray, np.ndarray],
    phis_psis_pred: tuple[np.ndarray, np.ndarray],
):
    angles_true = np.concatenate(phis_psis_true, axis=1)
    angles_pred = np.concatenate(phis_psis_pred, axis=1)
    T_W2 = _get_torus_wasserstein(angles_true, angles_pred)
    return T_W2


if __name__ == "__main__":
    from boltzkit.targets.boltzmann import MolecularBoltzmann
    from boltzkit.utils.pdf import save_pdf
    from boltzkit.evaluation.sample_based.visualize import (
        visualize_histogram_1d,
        visualize_histogram_2d,
    )

    bm = MolecularBoltzmann("datasets/chrklitz99/test_system")

    topology = bm.get_mdtraj_topology()

    gt_samples_path = bm._repo.load_file("300K_val.npy")
    gt_samples = np.load(gt_samples_path)

    angles = get_torsion_angles(gt_samples, topology)
    torsion_marginals = get_torsion_marginal_hists(*angles)

    for i, marginals_i in enumerate(torsion_marginals):
        visualize_histogram_2d(
            marginals_i["phi_psi"],
            show=True,
            xlabel=f"$\\phi_{i}$",
            ylabel=f"$\\psi_{i}$",
        )
        for angle_name in ("phi", "psi"):
            visualize_histogram_1d(
                marginals_i[angle_name],
                show=True,
                xlabel=f"$\\{angle_name}_{i}$",
            )
    # save_pdf(pdf_buffer, "ram.pdf")
