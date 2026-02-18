from dataclasses import dataclass

import numpy as np


@dataclass
class Histogram1D:
    counts: np.ndarray
    bin_edges: np.ndarray

    def get_bin_centers(self):
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])


@dataclass
class Histogram2D:
    counts: np.ndarray
    bin_edges_x: np.ndarray
    bin_edges_y: np.ndarray

    def get_extend(self):
        return [
            self.bin_edges_x[0],
            self.bin_edges_x[-1],
            self.bin_edges_y[0],
            self.bin_edges_y[-1],
        ]


def get_histogram_1d(
    data: np.ndarray,
    n_bins: int = 100,
    data_range: tuple[float, float] | None = None,
    density: bool = True,
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
    density: bool
        Whether to normalize counts or not, default is True.

    Returns
    -------
    Histogram1D
        A `Histogram1D` object containing the histogram and bin edges.
    """
    if data.ndim != 1:
        raise ValueError("samples must have shape (N,)")

    if data_range is None:
        data_range = (float(np.min(data)), float(np.max(data)))

    hist, bin_edges = np.histogram(data, bins=n_bins, range=data_range, density=density)
    return Histogram1D(hist, bin_edges)


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
    return Histogram2D(hist, xedges, yedges)
