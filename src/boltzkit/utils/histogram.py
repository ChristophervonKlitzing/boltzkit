import os
from typing import Protocol
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from boltzkit.utils.pdf import matplotlib_to_pdf_buffer
from boltzkit.utils.molecular.conversion import to_free_energy
from boltzkit.utils.shape_utils import get_balanced_grid


class Histogram1D:
    def __init__(
        self, counts: np.ndarray, bin_edges: np.ndarray, n_producing_samples: int
    ):
        self._normalized_counts: np.ndarray = counts / counts.sum()
        self._n_producing_samples = n_producing_samples
        self._bin_edges = bin_edges

    def __repr__(self):
        p = self.get_normalized_counts()
        x = self.get_bin_centers()
        mean = (p * x).sum()
        std = np.sqrt(((x - mean) ** 2 * p).sum())
        return f"Histogram1D(mean={mean:.2f},std={std:.2f})"

    def save(self, fpath: str):
        np.savez(
            fpath,
            #
            counts=self._normalized_counts,
            n_producing_samples=self.n_producing_samples,
            bin_edges=self._bin_edges,
            #
            allow_pickle=False,
        )

    @property
    def n_producing_samples(self) -> int:
        return self._n_producing_samples

    @property
    def bin_edges(self):
        return self._bin_edges

    def get_bin_centers(self):
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    def get_normalized_counts(self) -> np.ndarray:
        # normalize again just in case
        return self._normalized_counts / self._normalized_counts.sum()

    def get_approximate_absolute_counts(self) -> np.ndarray:
        return self.get_normalized_counts() * self.n_producing_samples

    def get_extend(self) -> tuple[float, float]:
        return [
            self.bin_edges[0],
            self.bin_edges[-1],
        ]

    def get_as_density(self) -> np.ndarray:
        counts = self.get_normalized_counts()
        return counts / self.get_bin_area()

    def get_bin_area(self):
        min_x, max_x = self.get_extend()
        x_range = max_x - min_x
        n_bins = self.get_num_bins()

        bin_width = x_range / n_bins
        return bin_width

    def get_num_bins(self) -> int:
        return self._normalized_counts.shape[0]


class Histogram2D:
    def __init__(
        self,
        counts: np.ndarray,
        bin_edges_x: np.ndarray,
        bin_edges_y: np.ndarray,
        n_producing_samples: int,
    ):
        self._normalized_counts: np.ndarray = counts / counts.sum()
        self._n_producing_samples = n_producing_samples
        self._bin_edges_x = bin_edges_x
        self._bin_edges_y = bin_edges_y

    def get_bin_centers(self):
        x_centers = 0.5 * (self._bin_edges_x[:-1] + self._bin_edges_x[1:])
        y_centers = 0.5 * (self._bin_edges_y[:-1] + self._bin_edges_y[1:])
        return x_centers, y_centers

    def __repr__(self):
        counts = self.get_normalized_counts()
        x_centers, y_centers = self.get_bin_centers()

        # Weighted means
        mean_x = (counts.sum(axis=1) * x_centers).sum()
        mean_y = (counts.sum(axis=0) * y_centers).sum()

        # Weighted stds
        std_x = np.sqrt(((counts.sum(axis=1) * (x_centers - mean_x) ** 2).sum()))
        std_y = np.sqrt(((counts.sum(axis=0) * (y_centers - mean_y) ** 2).sum()))

        return f"Histogram2D(mean=({mean_x:.2f},{mean_y:.2f}),std=({std_x:.2f},{std_y:.2f}))"

    def save(self, fpath: str):
        np.savez(
            fpath,
            #
            counts=self._normalized_counts,
            n_producing_samples=self.n_producing_samples,
            bin_edges_x=self._bin_edges_x,
            bin_edges_y=self._bin_edges_y,
            #
            allow_pickle=False,
        )

    @property
    def n_producing_samples(self) -> int:
        return self._n_producing_samples

    @property
    def bin_edges_x(self):
        return self._bin_edges_x

    @property
    def bin_edges_y(self):
        return self._bin_edges_y

    def get_approximate_absolute_counts(self) -> np.ndarray:
        return self.get_normalized_counts() * self.n_producing_samples

    def get_extend(self) -> tuple[float, float, float, float]:
        return [
            self.bin_edges_x[0],
            self.bin_edges_x[-1],
            self.bin_edges_y[0],
            self.bin_edges_y[-1],
        ]

    def get_bin_area(self):
        min_x, max_x, min_y, max_y = self.get_extend()
        x_range = max_x - min_x
        y_range = max_y - min_y
        n_bins_x, n_bins_y = self.get_num_bins()

        bin_width = x_range / n_bins_x
        bin_height = y_range / n_bins_y
        return bin_width * bin_height

    def get_normalized_counts(self) -> np.ndarray:
        # normalize again just in case
        return self._normalized_counts / self._normalized_counts.sum()

    def get_as_density(self) -> np.ndarray:
        counts = self.get_normalized_counts()
        return counts / self.get_bin_area()

    def get_num_bins(self) -> tuple[int, int]:
        return self._normalized_counts.shape


def get_histogram_1d(
    data: np.ndarray,
    n_bins: int = 100,
    data_range: tuple[float, float] | None = None,
):
    """
    Compute a 1D histogram of data.

    Parameters
    ----------
    data : np.ndarray
        1D array of shape (N,)
    n_bins : int, optional
        Number of histogram bins (default is 100).
    data_range : tuple[float, float] or None, optional
        Tuple specifying (min, max) range of the histogram.
        If None, the range is inferred from the data.

    Returns
    -------
    Histogram1D
        A `Histogram1D` object containing the histogram and bin edges.
    """
    if data.ndim != 1:
        raise ValueError("samples must have shape (N,)")

    if data_range is None:
        data_range = (float(np.min(data)), float(np.max(data)))

    hist, bin_edges = np.histogram(data, bins=n_bins, range=data_range, density=True)
    return Histogram1D(hist, bin_edges, data.shape[0])


def get_histogram_2d(
    data: np.ndarray,
    n_bins: int = 100,
    data_range: tuple[tuple[float, float], tuple[float, float]] | None = None,
    density: bool = True,
):
    """
    Compute and display a 2D Ramachandran histogram.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N, 2)
    n_bins : int, optional
        Number of bins per dimension.
    data_range : ((float, float), (float, float)) or None
        2D tuple specifying (min, max) range of the histogram.
        If None, the range is inferred from the data.
    density: bool
        Whether to normalize counts or not, default is True.

    Returns
    -------
    Histogram2D
        A `Histogram2D` object containing the histogram counts and bin edges.
    """

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("samples must have shape (N, 2)")

    data_x = data[:, 0]
    data_y = data[:, 1]

    if data_range is None:
        data_range = (
            (float(np.min(data_x)), float(np.max(data_x))),
            (float(np.min(data_y)), float(np.max(data_y))),
        )

    hist, xedges, yedges = np.histogram2d(
        data_x, data_y, bins=n_bins, range=data_range, density=density
    )
    return Histogram2D(hist, xedges, yedges, data.shape[0])


def load_histogram(fpath: str) -> Histogram1D | Histogram2D:
    data = np.load(fpath, allow_pickle=False)
    dict_data: dict[str, np.ndarray] = {k: data[k] for k in data}
    assert "count" in dict_data, "File does not seem to contain a histogram"

    dim = len(dict_data["count"].shape)
    if dim == 1:
        h = Histogram1D(**dict_data)
    elif dim == 2:
        h = Histogram2D(**dict_data)
    else:
        raise ValueError(f"Coult not load histogram")

    return h


def save_histograms(hists: dict[str, Histogram1D | Histogram2D], dirpath: str):
    for name, h in hists.items():
        fpath = os.path.join(dirpath, name)
        h.save(fpath)


class VisualizationMode(Protocol):
    @property
    def id(self) -> str:
        pass

    def __call__(self, hist: Histogram1D | Histogram2D) -> tuple[np.ndarray, str]:
        pass


def plot_as_free_energy(hist: Histogram1D | Histogram2D) -> tuple[np.ndarray, str]:
    y = to_free_energy(hist.get_normalized_counts())
    ylabel = r"free energy / $k_B T$"
    return y, ylabel


plot_as_free_energy.id = "free_energy"


def plot_as_density(hist: Histogram1D | Histogram2D) -> tuple[np.ndarray, str]:
    y = hist.get_as_density()

    # This leaves areas without samples empty (no information).
    y[y == 0] = np.nan

    ylabel = "probability density"
    return y, ylabel


plot_as_density.id = "density"


def plot_as_log_density(hist: Histogram1D | Histogram2D) -> tuple[np.ndarray, str]:
    density = hist.get_as_density()
    ylabel = "log density"

    mask = density > 0
    density[~mask] = 1e-300

    # Compute free energy for nonzero probabilities
    log_density = np.where(mask, np.log(density), -np.inf)

    return log_density, ylabel


plot_as_log_density.id = "log_density"


def visualize_histogram_1d(
    hist: Histogram1D,
    vis_mode: VisualizationMode = plot_as_log_density,
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
    vis_mode : VisualizationMode
        Transforms the given histogram counts/density into an np.ndarray for visualization.
        The function further returns default label, which is used if `ylabel` is None.
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
    pdf_buffer : PdfBuffer | None
        Buffer containing the generated PDF or None if `ax` is not None
    """

    # Compute bin centers
    x = hist.get_bin_centers()

    y, default_label = vis_mode(hist)
    if ylabel is None:
        ylabel = default_label

    # Create axes if not provided
    new_plot = ax is None
    if new_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if transpose:
        x, y = (y, x)
        xlabel, ylabel = (ylabel, xlabel)

    n_nans = np.isnan(y).astype(np.float32).sum()
    if n_nans == y.shape[0] - 1:  # all but one nan
        dirac_like = True
    else:
        dirac_like = False

    if dirac_like:
        dirac_idx = (~np.isnan(y)).astype(np.float32).argmax()
        offset = hist.get_bin_area() / 2
        x_min = x[dirac_idx] - offset
        x_max = x[dirac_idx] + offset
        y_dirac_vals = [0, y[dirac_idx], 0]
        x_dirac_vals = [x_min, x[dirac_idx], x_max]
        ax.plot(x_dirac_vals, y_dirac_vals, label=label, linestyle=linestyle)
    else:
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

    fig.tight_layout()

    if ax is None:
        pdf_buffer = matplotlib_to_pdf_buffer(fig)
    else:
        pdf_buffer = None

    # Show plot immediately if requested
    if show:
        plt.show()
    elif new_plot:
        plt.close(fig)

    return pdf_buffer


def visualize_histogram_2d(
    hist: Histogram2D,
    vis_mode: VisualizationMode = plot_as_log_density,
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

    vis_mode : VisualizationMode
        Transforms the given histogram counts/density into an np.ndarray for visualization.
        The function further returns default label, which is used if `cbar_label` is None.

    show : bool
        Whether to immediately display the plot.

    ax : plt.Axes, optional
        Axes to plot on; if None, a new figure/axes is created.

    Returns
    -------
    pdf_buffer : PdfBuffer
        Buffer containing the generated PDF.
    """

    z, default_label = vis_mode(hist)
    if cbar_label is None:
        cbar_label = default_label

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


def visualize_histograms(
    hists: list[Histogram2D] | list[Histogram1D | dict[str, Histogram1D]],
    vis_mode: VisualizationMode = plot_as_log_density,
    grid_shape: tuple[int, int] | None = None,
    show: bool = False,
    progressbar_description: str = "",
    **kwargs,
):
    num_hists = len(hists)

    if grid_shape is None:
        n_rows, n_cols = get_balanced_grid(num_hists)
    else:
        n_rows, n_cols = grid_shape

    base_width = 3.3
    base_height = 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(base_width * n_cols, base_height * n_rows),
        squeeze=False,
    )

    for i in tqdm(range(num_hists), desc=progressbar_description):
        # print(f"Visualize histogram [{i+1}/{num_hists}]")
        row = i // n_cols
        col = i % n_cols
        ax: plt.Axes = axes[row, col]

        obj = hists[i]

        if isinstance(obj, dict):
            for key, h in obj.items():
                visualize_histogram_1d(h, vis_mode=vis_mode, ax=ax, label=key, **kwargs)
        elif isinstance(obj, Histogram1D):
            visualize_histogram_1d(obj, vis_mode=vis_mode, ax=ax, **kwargs)
        else:
            visualize_histogram_2d(obj, vis_mode=vis_mode, ax=ax, **kwargs)

    for i in range(num_hists, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax: plt.Axes = axes[row, col]
        ax.set_visible(False)

    pdf_buffer = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    else:
        plt.close()

    return pdf_buffer
