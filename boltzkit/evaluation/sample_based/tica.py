from matplotlib import pyplot as plt
import numpy as np
from boltzkit.utils.histogram import (
    Histogram2D,
    VisualizationMode,
    get_histogram_2d,
    plot_as_log_density,
)
from boltzkit.utils.pdf import matplotlib_to_pdf_buffer
from boltzkit.evaluation.sample_based.wasserstein import get_euclidean_wasserstein_1_2
from boltzkit.utils.histogram import visualize_histogram_2d


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
    vis_mode: VisualizationMode = plot_as_log_density,
    ax: plt.Axes | None = None,
    show: bool = False,
):
    create_ax = ax is None
    if create_ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    visualize_histogram_2d(tica_hist, vis_mode=vis_mode, ax=ax)

    pdf = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    elif create_ax:
        plt.close()

    return pdf


def visualize_tica_true_and_pred(
    tica_hist_true: Histogram2D,
    tica_hist_pred: Histogram2D,
    vis_mode: VisualizationMode = plot_as_log_density,
    show: bool = False,
):
    fig, axes = plt.subplots(ncols=2, figsize=(7, 3))

    visualize_tica(tica_hist_true, vis_mode=vis_mode, ax=axes[0])
    axes[0].set_title("True")

    visualize_tica(tica_hist_pred, vis_mode=vis_mode, ax=axes[1])
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

    gt_samples = bm.load_dataset(T=300.0, type="val")

    tica_proj = tica_model.project_from_cartesian(gt_samples, topology)
    tica_hist = get_tica_hist(tica_proj)
    visualize_histogram_2d(tica_hist, show=True, vis_mode=plot_as_log_density)

    tica_proj_true = tica_proj[:1000]
    tica_proj_pred = tica_proj[:1000] + 0.01 * np.random.randn(*tica_proj_true.shape)
    _, tica_w2 = get_tica_wasserstein_1_2(tica_proj_true, tica_proj_pred)
    print("TICA W2:", tica_w2)
