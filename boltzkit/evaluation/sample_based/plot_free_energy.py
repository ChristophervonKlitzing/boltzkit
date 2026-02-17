"""
This implementation is copied and adapted from
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

from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np


def to_free_energy(hist: np.ndarray, shift_min: bool = False) -> np.ndarray:
    """
    Convert histogram counts into free energy values (in units of kT).

    Parameters
    ----------
    hist : np.ndarray
        Histogram counts.
    shift_min : bool, optional
        If True, shifts the minimum free energy to zero.

    Returns
    -------
    np.ndarray
        Free energy values corresponding to histogram counts.
    """
    # This function reproduces the behavior of PyEMMA's _to_free_energy in a self-contained form
    # to avoid the additional dependency.
    # This function performs a generic mathematical transformation from histogram counts to free energies.
    # Being independently implemented and a generic mathematical idea, PyEMMA's license does not apply to this code.

    probs = hist / hist.sum()

    # Prevent divide-by-zero warnings from np.log by replacing zeros with a tiny positive value.
    # Zero entries correspond to infinite free energy, so the exact replacement value does not matter.
    mask = probs > 0
    probs[~mask] = 1e-300

    # Compute free energy for nonzero probabilities
    fe = np.where(mask, -np.log(probs), np.inf)

    if shift_min:
        fe -= np.min(fe[np.isfinite(fe)])

    return fe


def plot_1D_marginal(
    xs: np.ndarray,
    weights: np.ndarray | None = None,
    plot_as_free_energy: bool = False,
    ax: plt.Axes | None = None,
    n_bins: int = 100,
    label: str | None = None,
    linestyle: str = "-",
    return_data: bool = False,
):
    hist, edges = np.histogram(xs, bins=n_bins, density=True, weights=weights)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    if plot_as_free_energy:
        nonzero = hist > 0
        free_energy = np.inf * np.ones(shape=hist.shape)
        free_energy[nonzero] = -np.log(hist[nonzero])
        free_energy[nonzero] -= np.min(free_energy[nonzero])

        ax.plot(
            (edges[:-1] + edges[1:]) / 2, free_energy, label=label, linestyle=linestyle
        )
    else:
        ax.plot((edges[:-1] + edges[1:]) / 2, hist, label=label, linestyle=linestyle)

    if return_data:
        return (edges[:-1] + edges[1:]) / 2, (
            hist if not plot_as_free_energy else free_energy
        )


def _get_histogram(
    xall: np.ndarray,
    yall: np.ndarray,
    nbins: int | Sequence[int],
    weights: np.ndarray = None,
    range: np.ndarray | None = None,
):
    """Compute a two-dimensional histogram.

    Args:
        xall: Sample x-coordinates.
        yall: Sample y-coordinates.
        nbins: Number of histogram bins used in each dimension.
        weights: Sample weights. If None, all samples have the same weight.
        range: The leftmost and rightmost edges of the bins along each dimension [[xmin, xmax], [ymin, ymax]].

    Returns:
        x: The bins' x-coordinates in meshgrid format.
        y: The bins' y-coordinates in meshgrid format.
        z: Histogram counts in meshgrid format.
    """

    z, xedge, yedge = np.histogram2d(
        xall, yall, bins=nbins, weights=weights, range=range
    )
    x = 0.5 * (xedge[:-1] + xedge[1:])
    y = 0.5 * (yedge[:-1] + yedge[1:])

    return x, y, z.T  # transpose to match x/y-directions


def plot_2D_free_energy(
    xall: np.ndarray,
    yall: np.ndarray,
    weights: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    nbins: int | Sequence[int] = 100,
    min_energy_zero: bool = True,
    kT: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "nipy_spectral",
    cbar: bool = True,
    cbar_label: str = r"free energy / $kT$",
    cax: plt.Axes | None = None,
    cbar_orientation: Literal["horizontal", "vertical"] = "vertical",
    range: np.ndarray | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a two-dimensional free energy map using a histogram of
    scattered data.

    Args:
        xall: Sample x-coordinates.
        yall: Sample y-coordinates.
        weights: Sample weights. If None, all samples have the same weight.
        ax: The ax to plot to; if ax=None, a new ax (and fig) is created.
        nbins: Number of histogram bins used in each dimension.
        min_energy_zero: Shifts the energy minimum to zero.
        kT: The value of kT in the desired energy unit. By default, energies are
            computed in kT (setting 1.0). If you want to measure the energy in
            kJ/mol at 298 K, use kT=2.479 and change the cbar_label accordingly.
        vmin: Lowest free energy value to be plotted.
        vmax: Highest free energy value to be plotted.
        cmap: The color map to use.
        cbar: Plot a color bar.
        cbar_label: Colorbar label string; use None to suppress it.
        cax: Plot the colorbar into a custom axes object instead of stealing space
            from ax.
        cbar_orientation: Colorbar orientation; choose 'vertical' or 'horizontal'.
        range: The range of the histogram. If None, the range is computed from the data.

    Returns:
        fig: The figure in which the used ax resides.
        ax: The ax in which the map was plotted.
    """

    if min_energy_zero and vmin is None:
        vmin = 0.0

    x, y, z = _get_histogram(
        xall,
        yall,
        nbins=nbins,
        weights=weights,
        range=range,
    )
    f = to_free_energy(z, shift_min=min_energy_zero) * kT

    if vmax is not None:
        f[f >= vmax] = np.inf

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    map = ax.imshow(
        f,
        extent=[
            x.min() if range is None else range[0][0],
            x.max() if range is None else range[0][1],
            y.min() if range is None else range[1][0],
            y.max() if range is None else range[1][1],
        ],
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    if cbar:
        if cax is None:
            cbar_ = fig.colorbar(map, ax=ax, orientation=cbar_orientation)
        else:
            cbar_ = fig.colorbar(map, cax=cax, orientation=cbar_orientation)
        if cbar_label is not None:
            cbar_.set_label(cbar_label)

    return fig, ax
