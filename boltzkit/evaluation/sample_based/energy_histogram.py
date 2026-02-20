from matplotlib import pyplot as plt
import numpy as np
from boltzkit.utils.histogram import (
    Histogram1D,
    get_histogram_1d,
    visualize_histogram_1d as _visualize_hist_1d,
)
from boltzkit.utils.pdf import matplotlib_to_pdf_buffer


def get_reduced_energy_hist(
    log_probs: np.ndarray,
    n_bins: int = 100,
    energy_range: tuple[float, float] | None = None,
    quantile_range: tuple[float, float] | None = (0.01, 0.99),
    margin_ratio: float = 0.5,
):
    """
    Compute a histogram of reduced energies from log-probabilities.

    The reduced energy (dimensionless) is defined as

        u(x) = -log p(x),

    i.e., the negative log-probability, where p can be unnormalized.

    For a Boltzmann distribution

        p(x) ∝ exp(-E(x) / (k_B T)),

    the reduced energy corresponds to

        u(x) = E(x) / (k_B T).

    Only relative differences in reduced energy are physically meaningful,
    since the normalization constant Z = ∫ p(x) dx is generally unknown.

    Parameters
    ----------
    log_probs : np.ndarray
        Array of log-probabilities log p(x). Can have arbitrary shape;
        the histogram is computed over all elements.

    n_bins : int, default=100
        Number of bins for the histogram.

    energy_range : tuple[float, float] | None, default=None
        Explicit (min, max) range for the histogram. If provided,
        `quantile_range` is ignored.

    quantile_range : tuple[float, float] | None, default=(0.01, 0.99)
        Lower and upper quantiles used to automatically determine the
        histogram range if `energy_range` is None. This is useful to
        exclude extreme outliers. If None, the full data range is used.

    margin_ratio : float, default=0.5
        Fraction of the data range to add as padding on both sides of the
        histogram range when it is determined from `quantile_range`. For
        example, a value of 0.5 extends the lower and upper bounds by
        50% of the selected quantile width, providing extra margin to
        ensure edge data points are included, which are not extreme outliers.

    Returns
    -------
        Histogram1D
        A Histogram1D object containing the histogram and bin edges
    """

    reduced_energy = -log_probs.flatten()

    # Determine histogram range
    if energy_range is None:
        if quantile_range is not None:
            q_low, q_high = quantile_range
            energy_range_mutable = [
                np.quantile(reduced_energy, q_low),
                np.quantile(reduced_energy, q_high),
            ]
            width = energy_range_mutable[1] - energy_range_mutable[0]
            energy_range_mutable[0] = energy_range_mutable[0] - width * margin_ratio
            energy_range_mutable[1] = energy_range_mutable[1] + width * margin_ratio

            energy_range = tuple(energy_range_mutable)
        else:
            energy_range = (reduced_energy.min(), reduced_energy.max())

    return get_histogram_1d(
        reduced_energy,
        n_bins=n_bins,
        data_range=energy_range,
    )


def visualize_energy_hist_dual(
    pred_energy_hist: Histogram1D, true_energy_hist: Histogram1D, show: bool = False
):
    fig, ax = plt.subplots()

    # Plot both on the same axis
    _visualize_hist_1d(
        true_energy_hist,
        ax=ax,
        label="True",
        show=False,
    )

    _visualize_hist_1d(
        pred_energy_hist,
        ax=ax,
        label="Pred",
        show=False,
    )

    ax.legend()
    # ax.set_title("Reduced Energy Histogram")
    ax.set_xlabel("energy")
    ax.set_ylabel("probability density")

    pdf_buffer = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    else:
        plt.close()

    return pdf_buffer


if __name__ == "__main__":

    # Generate 1D Gaussian samples
    rng = np.random.default_rng(seed=42)

    # dummy log probs
    log_probs = rng.normal(loc=0.0, scale=1.0, size=100_000)
    log_probs2 = rng.normal(loc=1.0, scale=1.3, size=100_000)

    # Test with extreme outliers
    log_probs[0] = -1e8
    log_probs[2] = 1e8

    # Compute reduced energy histogram using quantile selection
    energy_hist = get_reduced_energy_hist(log_probs=log_probs)
    energy_hist2 = get_reduced_energy_hist(log_probs=log_probs2)
    visualize_energy_hist_dual(energy_hist, energy_hist2, show=True)
