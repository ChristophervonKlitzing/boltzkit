import math
import matplotlib.pyplot as plt
import numpy as np

from boltzkit.utils.pdf import matplotlib_to_pdf_buffer
from .molecular.conversion import to_free_energy
from .histogram import Histogram1D, Histogram2D


def visualize_histogram_1d(
    hist: Histogram1D,
    plot_as_free_energy: bool = False,
    show: bool = False,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    label: str | None = None,
    linestyle: str = "-",
    transpose: bool = False,
):
    """
    Plot a 1D histogram as counts or free energy.

    Parameters
    ----------
    hist : np.ndarray
        Histogram counts or probabilities. If plotting as free energy, these
        will be normalized internally. Shape: (n_bins,)

    bin_edges : np.ndarray
        Bin edges defining the histogram intervals. Length is one greater
        than the number of bins, i.e., shape: (n_bins + 1,). The i-th bin
        covers the interval [bin_edges[i], bin_edges[i+1]).
    plot_as_free_energy : bool
        Whether to convert counts to free energy (-ln P).
    show : bool
        Whether to immediately display the plot.
    ax : plt.Axes, optional
        Axes to plot on; if None, a new figure/axes is created.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    label : str, optional
        Legend label.
    linestyle : str
        Line style for the plot.

    Returns
    -------
    pdf_buffer : io.BytesIO
        Buffer containing the generated PDF.
    """

    # Compute bin centers
    x = hist.get_bin_centers()

    # Convert to free energy if requested
    if plot_as_free_energy:
        y = to_free_energy(hist.get_normalized_counts())
        if ylabel is None:
            ylabel = r"free energy / $k_B T$"
    else:
        y = hist.get_as_density()

        # This leaves areas without samples empty (no information).
        y[y == 0] = np.nan

        if ylabel is None:
            ylabel = "probability density"

    # Create axes if not provided
    new_plot = ax is None
    if new_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if transpose:
        x, y = (y, x)
        xlabel, ylabel = (ylabel, xlabel)

    # Plot the data
    ax.plot(x, y, label=label, linestyle=linestyle)

    # Add optional titles and labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if label:
        ax.legend()

    # ax.yaxis.set_label_position(ylabel_postion)

    fig.tight_layout()
    pdf_buffer = matplotlib_to_pdf_buffer(fig)

    # Show plot immediately if requested
    if show:
        plt.show()
    elif new_plot:
        plt.close(fig)

    return pdf_buffer


def visualize_histogram_2d(
    hist: Histogram2D,
    plot_as_free_energy: bool = False,
    show: bool = False,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: None | str = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar: bool = True,
    cbar_label: str | None = None,
):
    """
    Plot a 2D histogram as counts or free energy and return a PDF buffer.

    Parameters
    ----------
    hist : np.ndarray
        2D histogram counts or probabilities. Shape: (nx, ny)

    x_bin_edges : np.ndarray
        Bin edges for x-dimension. Shape: (nx + 1,)

    y_bin_edges : np.ndarray
        Bin edges for y-dimension. Shape: (ny + 1,)

    plot_as_free_energy : bool
        Whether to convert histogram to free energy (-ln P).

    show : bool
        Whether to immediately display the plot.

    ax : plt.Axes, optional
        Axes to plot on; if None, a new figure/axes is created.

    Returns
    -------
    pdf_buffer : io.BytesIO
        Buffer containing the generated PDF.
    """

    # Convert to free energy if requested
    if plot_as_free_energy:
        z = to_free_energy(hist.get_normalized_counts())
        if cbar_label is None:
            cbar_label = r"free energy / $k_B T$"
    else:
        z = hist.get_as_density()

        # This makes areas without samples white (no information),
        # which increases contrast in low-density (but not zero) regions.
        z[z == 0] = np.nan

        if cbar_label is None:
            cbar_label = "probability density"

    # Figure ownership
    new_plot = ax is None
    if new_plot:
        if cbar:
            figsize = (6, 5)
        else:
            figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot
    im = ax.imshow(
        z.T,
        extent=hist.get_extend(),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # Labels and title
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Colorbar
    if cbar:
        cbar_obj = fig.colorbar(im, ax=ax)
        if cbar_label is not None:
            cbar_obj.set_label(cbar_label)

    fig.tight_layout()

    pdf_buffer = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    elif new_plot:
        plt.close(fig)

    return pdf_buffer


def get_balanced_subplot_grid(n: int) -> tuple[int, int]:
    """
    Given n subplots, returns a (rows, cols) tuple for a balanced layout.

    - Tries to make it roughly square.
    - If not square, prefers one row less than columns.
    """
    if n <= 0:
        raise ValueError("Number of subplots must be positive")

    # Start with the integer square root
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # If rows == cols - 1, good; if rows < cols - 1, try to reduce cols
    if rows < cols - 1:
        cols -= 1
        rows = math.ceil(n / cols)

    return rows, cols


if __name__ == "__main__":
    for i in range(1, 15):
        rows, cols = get_balanced_subplot_grid(i)
        print(i, rows, cols, rows * cols)
