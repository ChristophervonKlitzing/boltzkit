from collections import defaultdict
from typing import Protocol, TypeVar

import numpy as np

from boltzkit.utils.histogram import Histogram1D, Histogram2D

T = TypeVar("T", Histogram1D, Histogram2D)


class HistogramMetric(Protocol):
    """
    Protocol for histogram distance/divergence metrics.

    A histogram metric is a callable that takes two histograms of the same
    dimensionality and binning and returns a scalar value describing the
    difference between them.

    Implementations must also provide an ``id`` property that uniquely
    identifies the metric (used for logging and aggregation).

    Methods
    -------
    __call__(hist_p, hist_q) -> float
        Compute the metric between two histograms.

    Properties
    ----------
    id : str
        Short identifier for the metric (e.g., "kl", "tv").
    """

    def __call__(self, hist_p: T, hist_q: T) -> float:
        pass

    @property
    def id(self) -> str:
        pass


def get_histogram_fwd_kullback_leibler(hist_p: T, hist_q: T):
    """
    Compute the forward Kullback-Leibler divergence between two histograms.

    The divergence is computed as

        KL(P || Q) = ∫ P(x) log(P(x) / Q(x)) dx

    using the density representation of the histograms and approximating
    the integral via a discrete sum over bins.

    A small epsilon (1e-10) is added to both densities to avoid numerical issues
    when taking the logarithm.

    Parameters
    ----------
    hist_p : Histogram1D or Histogram2D
        Reference histogram representing distribution P.
    hist_q : Histogram1D or Histogram2D
        Comparison histogram representing distribution Q.
        Must have the same number of bins as ``hist_p``.

    Returns
    -------
    float
        The forward Kullback-Leibler divergence KL(P || Q).

    Notes
    -----
    - Both histograms must have identical binning.
    - The bin area is used to correctly approximate the continuous integral.
    """
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


get_histogram_fwd_kullback_leibler.id = "kl"


def get_histogram_total_variation_distance(hist_p: T, hist_q: T):
    """
    Compute the total variation distance between two histograms.

    The total variation distance between two probability densities is

        TV(P, Q) = 1/2 ∫ |P(x) - Q(x)| dx

    The integral is approximated via a discrete sum over the histogram bins.

    Parameters
    ----------
    hist_p : Histogram1D or Histogram2D
        First histogram representing distribution P.
    hist_q : Histogram1D or Histogram2D
        Second histogram representing distribution Q.
        Must have the same number of bins as ``hist_p``.

    Returns
    -------
    float
        The total variation distance between the two distributions.

    Notes
    -----
    - Both histograms must have identical binning.
    - The bin area is used to properly approximate the integral.
    """
    assert hist_p.get_num_bins() == hist_q.get_num_bins()

    hist_ram_p = hist_p.get_as_density()
    hist_ram_q = hist_q.get_as_density()

    bin_area = hist_p.get_bin_area()

    total_variation_ram = (
        0.5 * np.sum(np.abs(hist_ram_p - hist_ram_q)) * bin_area
    )  # To get the properly normalized integral / KLD
    return float(total_variation_ram)


get_histogram_total_variation_distance.id = "tv"


T = TypeVar("T", Histogram1D, Histogram2D)


def get_histogram_metrics(
    hist_metrics: list[HistogramMetric],
    true: list[T],
    pred: list[T],
    *,
    group: str,
    h_type: str | None = None,
    include_individual: bool = True,
    include_aggregated: bool = True,
):
    """
    Evaluate multiple histogram metrics for pairs of histograms.

    Each metric is computed for corresponding pairs of histograms from
    ``true`` and ``pred``. Results are returned in a flat dictionary with
    keys encoding the group, histogram-type, metric identifier, and histogram index.

    Additionally, the mean value of each metric across all histogram pairs
    is computed and stored.

    Parameters
    ----------
    hist_metrics : list of HistogramMetric
        Metrics to evaluate on each histogram pair.
    true : list of Histogram1D or Histogram2D
        List of reference ("ground truth") histograms.
    pred : list of Histogram1D or Histogram2D
        List of predicted histograms corresponding to ``true``.
    group : str
        Name of the metric group (e.g., `torsion_marginal`).
    h_type : str
        Descriptor for the histogram type (e.g., "phi_psi" (2D), "phi" (1D)).
    include_individual : bool, default=True
        Whether to include metric values for each individual histogram pair.
    include_aggregated : bool, default=True
        Whether to include aggregated metrics across all histogram pairs
        (currently the mean for each metric).

    Returns
    -------
    dict[str, float]
        Dictionary mapping metric names to computed values.

        Individual metric entries follow the format::

            "{group}/{h_type}_{metric_id}_{i}"

            or

            "{group}/{metric_id}_{i}" if `h_type` is `None`

        where ``i`` is the histogram index.

        Mean values across all histogram pairs are stored as::

            "{group}/{h_type}_{metric_id}_mean"

            or

            "{group}/{metric_id}_{i}" if `h_type` is `None`

    Notes
    -----
    - ``true`` and ``pred`` must contain the same number of histograms.
    - Histogram pairs are evaluated in order using ``zip(true, pred)``.
    """

    assert include_aggregated or include_individual

    metrics = {}
    summed_metrics: dict[str, list[float]] = defaultdict(list)
    _type = f"{h_type}_" if h_type is not None else ""

    # Iterate over the torsion backbone angle pairs
    for i, (true_hist, pred_hist) in enumerate(zip(true, pred)):
        # evaluate all specified metrics on the angle histogram pairs
        for metric in hist_metrics:
            m_i = metric(true_hist, pred_hist)

            if include_individual:
                metrics[f"{group}/{_type}{metric.id}_{i}"] = m_i
            summed_metrics[metric.id].append(m_i)

    if include_aggregated:
        for id, metric_values in summed_metrics.items():
            metrics[f"{group}/{_type}{id}_mean"] = np.array(metric_values).mean()

    return metrics
