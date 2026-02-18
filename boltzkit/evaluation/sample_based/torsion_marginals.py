from typing import Literal, TypeAlias
import numpy as np

from boltzkit.utils.molecular.marginals import get_phi_psi_vectors
import mdtraj as md
from boltzkit.utils.histogram import (
    Histogram1D,
    Histogram2D,
    get_histogram_1d,
    get_histogram_2d,
)
from .wasserstein import get_torus_wasserstein as _get_torus_wasserstein

T: TypeAlias = dict[Literal["phi_psi", "phi", "psi"], Histogram1D | Histogram2D]


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


def get_torsion_marginal_hists(phis: np.ndarray, psis: np.ndarray, **kwargs) -> list[T]:
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
    list[T]
        A list of dictionaries (one per torsion pair). Each dictionary has keys:
        - "phi_psi": 2D `Histogram2D` of the joint φ/ψ angles
        - "phi": 1D `Histogram1D` of φ angles
        - "psi": 1D `Histogram1D` of ψ angles
    """
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


if __name__ == "__main__":
    from boltzkit.targets.boltzmann import MolecularBoltzmann
    from boltzkit.utils.pdf import save_pdf
    from boltzkit.utils.visualize import (
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
        ram_hist = marginals_i["phi_psi"]

        adjusted_counts = ram_hist.get_as_density() + 0.001 * np.abs(
            np.random.randn(*ram_hist.get_num_bins())
        )
        fake_ram_hist = Histogram2D(
            adjusted_counts,
            ram_hist.bin_edges_x,
            ram_hist.bin_edges_y,
            ram_hist.n_producing_samples,
        )
        ram_kl = get_ramachandran_kl(ram_hist, fake_ram_hist)
        ram_tv = get_ramachandran_total_variation(ram_hist, fake_ram_hist)
        print(f"Fake Ram KL: {ram_kl:.4f}")
        print(f"Fake Ram TV: {ram_tv:.4f}")

        visualize_histogram_2d(
            ram_hist,
            show=True,
            xlabel=f"$\\phi_{i}$",
            ylabel=f"$\\psi_{i}$",
            plot_as_free_energy=True,
        )
        visualize_histogram_2d(
            fake_ram_hist,
            show=True,
            xlabel=f"$\\phi_{i}$",
            ylabel=f"$\\psi_{i}$",
            plot_as_free_energy=True,
        )
        for angle_name in ("phi", "psi"):
            visualize_histogram_1d(
                marginals_i[angle_name],
                show=True,
                xlabel=f"$\\{angle_name}_{i}$",
                plot_as_free_energy=True,
            )
    # save_pdf(pdf_buffer, "ram.pdf")
