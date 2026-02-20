from matplotlib import pyplot as plt
import numpy as np
import mdtraj as md
import deeptime as dt
from boltzkit.utils.histogram import Histogram2D, get_histogram_2d
from boltzkit.utils.pdf import matplotlib_to_pdf_buffer
from .wasserstein import get_euclidean_wasserstein_1_2
from ._tica_help import _tica_features
from boltzkit.utils.histogram import visualize_histogram_2d


def get_tica_projections(
    data: np.ndarray,
    topology: md.Topology,
    tica_model: dt.decomposition.TransferOperatorModel,
) -> np.ndarray:
    """
    Project molecular trajectory data onto a trained TICA model.

    Parameters
    ----------
    data : np.ndarray
        Array of molecular coordinates of shape (batch, num_atoms*3) or (batch, num_atoms, 3),
        where `batch` is the number of frames, and the last dimension corresponds to (x, y, z) coordinates.
    topology : md.Topology
        MDTraj topology describing the molecular system.
    tica_model : deeptime.decomposition.TransferOperatorModel
        Trained TICA model to transform features into TICA space.

    Returns
    -------
    np.ndarray
        TICA projections of shape (batch, n_tica_components),
        where `n_tica_components` is the number of components in the TICA model.
    """
    data = data.reshape(data.shape[0], -1, 3)
    ticas_list = []
    for i in range(0, data.shape[0], 50000):
        tica_features_batch = _tica_features(
            md.Trajectory(
                xyz=data[i : i + 50000],
                topology=topology,
            )
        )
        ticas_list.append(tica_model.transform(tica_features_batch))
    return np.concatenate(ticas_list, axis=0)


def get_tica_hist(
    tica_proj: np.ndarray,
    **kwargs,
):
    """
    Compute a 2D histogram of TICA projections.

    Parameters
    ----------
    tica_proj : np.ndarray
        Array of shape (batch, 2), containing the first two TICA components per frame.
    **kwargs
        Additional keyword arguments passed to `get_histogram_2d`.

    Returns
    -------
    Histogram2D
        A Histogram2D object containing the relative histogram counts and bin edges.
    """
    assert tica_proj.ndim == 2 and tica_proj.shape[1] == 2
    return get_histogram_2d(tica_proj, density=True, **kwargs)


def visualize_tica(
    tica_hist: Histogram2D,
    plot_as_free_energy: bool,
    ax: plt.Axes | None = None,
    show: bool = False,
):
    create_ax = ax is None
    if create_ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    visualize_histogram_2d(tica_hist, plot_as_free_energy=plot_as_free_energy, ax=ax)

    pdf = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    elif create_ax:
        plt.close()

    return pdf


def visualize_tica_true_and_pred(
    tica_hist_true: Histogram2D,
    tica_hist_pred: Histogram2D,
    plot_as_free_energy: bool = True,
    show: bool = False,
):
    fig, axes = plt.subplots(ncols=2, figsize=(7, 3))

    visualize_tica(tica_hist_true, plot_as_free_energy=plot_as_free_energy, ax=axes[0])
    axes[0].set_title("True")

    visualize_tica(tica_hist_pred, plot_as_free_energy=plot_as_free_energy, ax=axes[1])
    axes[1].set_title("Pred")

    pdf = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    else:
        plt.close()

    return pdf


def get_tica_wasserstein_1_2(
    tica_projections_true: np.ndarray,
    tica_projections_pred: np.ndarray,
    include_w1=False,
    include_w2=True,
):
    """
    Compute the Wasserstein distances (W1 and W2) between two sets of TICA projections.

    Parameters
    ----------
    tica_projections_true : np.ndarray
        Reference TICA projections of shape (batch, n_tica_components).
    tica_projections_pred : np.ndarray
        Predicted TICA projections of shape (batch, n_tica_components).
    include_w1 : bool, optional
        Whether to compute the W1 distance (default is False).
    include_w2 : bool, optional
        Whether to compute the W2 distance (default is True).

    Returns
    -------
    tuple
        Tuple `(W1, W2)` of Wasserstein distances. Each is either a float or `None`
        depending on `include_w1` and `include_w2`.
    """
    tica_W1, tica_W2 = get_euclidean_wasserstein_1_2(
        tica_projections_pred,
        tica_projections_true,
        include_w1=include_w1,
        include_w2=include_w2,
    )
    return tica_W1, tica_W2


if __name__ == "__main__":
    from boltzkit.targets.boltzmann import MolecularBoltzmann

    bm = MolecularBoltzmann("datasets/chrklitz99/test_system")

    topology = bm.get_mdtraj_topology()
    tica_model = bm.get_tica_model()

    gt_samples_path = bm._repo.load_file("300K_val.npy")
    gt_samples = np.load(gt_samples_path)

    tica_proj = get_tica_projections(gt_samples, topology, tica_model)
    tica_hist = get_tica_hist(tica_proj)
    visualize_histogram_2d(tica_hist, show=True, vmax=12.5, plot_as_free_energy=True)

    tica_proj_true = tica_proj[:1000]
    tica_proj_pred = tica_proj[:1000] + 0.01 * np.random.randn(*tica_proj_true.shape)
    _, tica_w2 = get_tica_wasserstein_1_2(tica_proj_true, tica_proj_pred)
    print("TICA W2:", tica_w2)
