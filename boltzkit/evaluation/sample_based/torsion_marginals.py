import numpy as np

from boltzkit.utils.molecular.marginals import get_phi_psi_vectors
import mdtraj as md
from boltzkit.utils.histogram import (
    Histogram1D,
    Histogram2D,
    get_histogram_1d,
    get_histogram_2d,
)
from boltzkit.utils.shape_utils import get_balanced_grid
from boltzkit.evaluation.sample_based.wasserstein import (
    get_torus_wasserstein as _get_torus_wasserstein,
)
from matplotlib import pyplot as plt


from boltzkit.utils.histogram import visualize_histogram_1d, visualize_histogram_2d


from boltzkit.utils.pdf import PdfBuffer, matplotlib_to_pdf_buffer


def get_torsion_angles(samples: np.ndarray, topology: md.Topology):
    """
    Extract backbone φ and ψ torsion angles in the range (-pi, pi) from a batch of molecular coordinates.

    Parameters
    ----------
    samples : np.ndarray
        Cartesian coordinates of shape (batch, n_atoms, 3) or (batch, n_atoms*3),
        where `batch` is the number of frames and the last dimension corresponds to (x, y, z).
    topology : md.Topology
        MDTraj topology describing the molecular system.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of arrays `(phis, psis)`:
        - phis : np.ndarray of shape (batch, n_torsions)
        - psis : np.ndarray of shape (batch, n_torsions)
        Each entry contains the backbone φ or ψ angles in radians.
    """
    # An alias (here for completeness)
    return get_phi_psi_vectors(samples, topology)


def get_torsion_marginal_hists(
    phis: np.ndarray, psis: np.ndarray, **kwargs
) -> tuple[list[Histogram2D], list[Histogram1D], list[Histogram1D]]:
    """
    Compute 1D and 2D histograms of torsion angles for all φ/ψ pairs.

    Parameters
    ----------
    phis : np.ndarray
        Array of shape (batch, n_torsions) with φ angles in radians in range (-pi, pi).
    psis : np.ndarray
        Array of shape (batch, n_torsions) with ψ angles in radians in range (-pi, pi).
    **kwargs
        Additional keyword arguments forwarded to `get_histogram_1d` and `get_histogram_2d`.

    Returns
    -------
    tuple[Histogram2D, Histogram1D, Histogram1D]
        A tuple of equal-size lists (one list element per torsion angle pair).
        - "phi_psi": 2D `Histogram2D` of the joint φ/ψ angles
        - "phi": 1D `Histogram1D` of φ angles
        - "psi": 1D `Histogram1D` of ψ angles
    """
    angles = np.stack([phis, psis], -1)  # (batch, <num-angle-pairs>, 2)

    range_1d = (-np.pi, np.pi)
    range_2d = ((-np.pi, np.pi), (-np.pi, np.pi))

    phi_psi_hists = []
    phi_hists = []
    psi_hists = []

    for angle_pair_idx in range(angles.shape[1]):

        # 2D histogram
        hist_2d = get_histogram_2d(
            angles[:, angle_pair_idx, :], data_range=range_2d, **kwargs
        )
        phi_psi_hists.append(hist_2d)

        # 2 1D histograms
        phi_hist_1d = get_histogram_1d(
            angles[:, angle_pair_idx, 0], data_range=range_1d, **kwargs
        )
        phi_hists.append(phi_hist_1d)

        psi_hist_1d = get_histogram_1d(
            angles[:, angle_pair_idx, 1], data_range=range_1d, **kwargs
        )
        psi_hists.append(psi_hist_1d)

    return phi_psi_hists, phi_hists, psi_hists


def get_torus_wasserstein_2(
    phis_psis_true: tuple[np.ndarray, np.ndarray],
    phis_psis_pred: tuple[np.ndarray, np.ndarray],
):
    """
    Compute the toroidal 2-Wasserstein distance between precomputed φ/ψ angles given in range (-pi, pi).

    Parameters
    ----------
    phis_psis_true : tuple[np.ndarray, np.ndarray]
        Tuple `(phis, psis)` for the reference ensemble:
        - phis: (batch, n_torsions)
        - psis: (batch, n_torsions)
    phis_psis_pred : tuple[np.ndarray, np.ndarray]
        Tuple `(phis, psis)` for the predicted ensemble:
        - phis: (batch, n_torsions)
        - psis: (batch, n_torsions)

    Returns
    -------
    float
        Toroidal 2-Wasserstein distance between the joint φ/ψ distributions.

    Notes
    -----
    - The joint angular distribution is treated as living on a torus, therefore respecting the periodicity.
    """
    angles_true = np.concatenate(phis_psis_true, axis=1)
    angles_pred = np.concatenate(phis_psis_pred, axis=1)
    T_W2 = _get_torus_wasserstein(angles_true, angles_pred)
    return T_W2


def get_ramachandran_kl(hist_p: Histogram2D, hist_q: Histogram2D):
    assert hist_p.get_num_bins() == hist_q.get_num_bins()

    hist_ram_p = hist_p.get_as_density()
    hist_ram_q = hist_q.get_as_density()

    eps_ram = 1e-10
    bin_area = hist_p.get_bin_area()

    log_ratio = np.log(hist_ram_p + eps_ram) - np.log(hist_ram_q + eps_ram)
    kld_ram = (
        np.sum(hist_ram_p * log_ratio)
        * bin_area  # To get the properly normalized integral / KLD
    )
    return float(kld_ram)


def get_ramachandran_total_variation(hist_p: Histogram2D, hist_q: Histogram2D):
    assert hist_p.get_num_bins() == hist_q.get_num_bins()

    hist_ram_p = hist_p.get_as_density()
    hist_ram_q = hist_q.get_as_density()

    bin_area = hist_p.get_bin_area()

    total_variation_ram = (
        0.5 * np.sum(np.abs(hist_ram_p - hist_ram_q)) * bin_area
    )  # To get the properly normalized integral / KLD
    return float(total_variation_ram)


def visualize_torsion_marginals_single_type(
    hists: list[Histogram2D] | list[Histogram1D],
    plot_as_free_energy: bool,
    grid_shape: tuple[int, int] | None = None,
    show: bool = False,
    **kwargs,
):
    num_hists = len(hists)

    if grid_shape is None:
        n_rows, n_cols = get_balanced_grid(num_hists)
    else:
        n_rows, n_cols = grid_shape

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)

    for i in range(num_hists):
        row = i // n_cols
        col = i % n_cols
        ax: plt.Axes = axes[row, col]

        h = hists[i]

        if isinstance(h, Histogram1D):
            visualize_histogram_1d(
                h, plot_as_free_energy=plot_as_free_energy, ax=ax, **kwargs
            )
        else:
            visualize_histogram_2d(
                h, plot_as_free_energy=plot_as_free_energy, ax=ax, **kwargs
            )

    pdf_buffer = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()

    return pdf_buffer


def visualize_torsion_marginals_per_type(
    torsion_marginals: tuple[list[Histogram2D], list[Histogram1D], list[Histogram1D]],
    plot_as_free_energy: bool,
    grid_shape: tuple[int, int] | None = None,
    show: bool = False,
    **kwargs,
) -> tuple[PdfBuffer, PdfBuffer, PdfBuffer]:
    labels = (("phi", "psi"), ("phi", None), (None, "psi"))
    pdf_list = []
    for h, labels in zip(torsion_marginals, labels):
        xlabel, ylabel = labels
        pdf = visualize_torsion_marginals_single_type(
            h,
            plot_as_free_energy=plot_as_free_energy,
            grid_shape=grid_shape,
            show=show,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )
        pdf_list.append(pdf)

    return tuple(pdf_list)


def visualize_torsion_marginals_all(
    torsion_marginals: tuple[list[Histogram2D], list[Histogram1D], list[Histogram1D]],
    plot_as_free_energy: bool,
    show: bool = False,
    **kwargs,
):
    assert len(torsion_marginals[0]) == len(torsion_marginals[1])
    assert len(torsion_marginals[1]) == len(torsion_marginals[2])

    n_pairs = len(torsion_marginals[0])

    fig, axes = plt.subplots(n_pairs, 3, squeeze=False, figsize=(10, 3 * n_pairs))
    for i in range(n_pairs):
        ax_ram: plt.Axes = axes[i, 2]
        ax_phi: plt.Axes = axes[i, 0]
        ax_psi: plt.Axes = axes[i, 1]

        h_ram = torsion_marginals[0][i]
        h_phi = torsion_marginals[1][i]
        h_psi = torsion_marginals[2][i]

        phi_label = f"$\\phi_{i}$"
        psi_label = f"$\\psi_{i}$"

        visualize_histogram_2d(
            h_ram,
            plot_as_free_energy=plot_as_free_energy,
            ax=ax_ram,
            xlabel=phi_label,
            ylabel=psi_label,
            **kwargs,
        )

        visualize_histogram_1d(
            h_phi,
            plot_as_free_energy=plot_as_free_energy,
            ax=ax_phi,
            xlabel=phi_label,
            **kwargs,
        )

        visualize_histogram_1d(
            h_psi,
            plot_as_free_energy=plot_as_free_energy,
            ax=ax_psi,
            xlabel=psi_label,
            **kwargs,
            transpose=False,
        )

    pdf = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    else:
        plt.close()

    return pdf


def visualize_torsion_marginals_dual(
    torsion_marginals_true: tuple[
        list[Histogram2D], list[Histogram1D], list[Histogram1D]
    ],
    torsion_marginals_pred: tuple[
        list[Histogram2D], list[Histogram1D], list[Histogram1D]
    ],
    plot_as_free_energy: bool,
    show: bool = False,
    **kwargs,
):
    assert len(torsion_marginals_true[0]) == len(torsion_marginals_true[1])
    assert len(torsion_marginals_true[1]) == len(torsion_marginals_true[2])

    assert len(torsion_marginals_pred[0]) == len(torsion_marginals_pred[1])
    assert len(torsion_marginals_pred[1]) == len(torsion_marginals_pred[2])

    n_pairs = len(torsion_marginals_true[0])

    fig, axes = plt.subplots(n_pairs, 4, squeeze=False, figsize=(13, 3 * n_pairs))
    for i in range(n_pairs):
        ax_ram_true: plt.Axes = axes[i, 2]
        ax_ram_pred: plt.Axes = axes[i, 3]
        ax_phi: plt.Axes = axes[i, 0]
        ax_psi: plt.Axes = axes[i, 1]

        h_ram_true = torsion_marginals_true[0][i]
        h_phi_true = torsion_marginals_true[1][i]
        h_psi_true = torsion_marginals_true[2][i]

        h_phi_pred = torsion_marginals_pred[1][i]
        h_ram_pred = torsion_marginals_pred[0][i]
        h_psi_pred = torsion_marginals_pred[2][i]

        phi_label = f"$\\phi_{i}$"
        psi_label = f"$\\psi_{i}$"

        visualize_histogram_2d(
            h_ram_true,
            plot_as_free_energy=plot_as_free_energy,
            ax=ax_ram_true,
            title="True",
            xlabel=phi_label,
            ylabel=psi_label,
            **kwargs,
        )

        visualize_histogram_2d(
            h_ram_pred,
            plot_as_free_energy=plot_as_free_energy,
            ax=ax_ram_pred,
            title="Pred",
            xlabel=phi_label,
            ylabel=psi_label,
            **kwargs,
        )

        visualize_histogram_1d(
            h_phi_true,
            plot_as_free_energy=plot_as_free_energy,
            ax=ax_phi,
            label="True",
            xlabel=phi_label,
            **kwargs,
        )
        visualize_histogram_1d(
            h_phi_pred,
            plot_as_free_energy=plot_as_free_energy,
            ax=ax_phi,
            label="Pred",
            xlabel=phi_label,
            **kwargs,
        )

        visualize_histogram_1d(
            h_psi_true,
            plot_as_free_energy=plot_as_free_energy,
            ax=ax_psi,
            label="True",
            xlabel=psi_label,
            **kwargs,
            transpose=False,
        )
        visualize_histogram_1d(
            h_psi_pred,
            plot_as_free_energy=plot_as_free_energy,
            ax=ax_psi,
            label="Pred",
            xlabel=psi_label,
            **kwargs,
            transpose=False,
        )

    pdf = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    else:
        plt.close()

    return pdf


if __name__ == "__main__":
    from boltzkit.targets.boltzmann import MolecularBoltzmann
    from boltzkit.utils.pdf import save_pdf
    from boltzkit.utils.histogram import (
        visualize_histogram_1d,
        visualize_histogram_2d,
    )

    bm = MolecularBoltzmann("datasets/chrklitz99/test_system")

    topology = bm.get_mdtraj_topology()

    gt_samples = bm.load_dataset(T=300.0, type="val")

    angles = get_torsion_angles(gt_samples, topology)
    angles2 = get_torsion_angles(
        gt_samples + 0.1 * np.random.randn(*gt_samples.shape), topology
    )

    torsion_marginals = get_torsion_marginal_hists(*angles)
    torsion_marginals2 = get_torsion_marginal_hists(*angles2)
    plot_as_free_energy = True

    pdf_buffer = visualize_torsion_marginals_dual(
        torsion_marginals,
        torsion_marginals2,
        plot_as_free_energy=plot_as_free_energy,
        show=True,
    )

    # save_pdf(pdf_buffer, "torsions.pdf")
