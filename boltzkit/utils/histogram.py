from abc import ABC
import os
import numpy as np


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
        return counts / self.get_bin_width()

    def get_bin_width(self):
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
