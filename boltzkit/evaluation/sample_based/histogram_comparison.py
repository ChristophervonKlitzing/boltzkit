from typing import Protocol, TypeVar

import numpy as np

from boltzkit.utils.histogram import Histogram1D, Histogram2D

T = TypeVar("T", Histogram1D, Histogram2D)


class HistogramMetric(Protocol):
    def __call__(self, hist_p: T, hist_q: T) -> float:
        pass

    @property
    def id(self) -> str:
        pass


def get_histogram_fwd_kullback_leibler(hist_p: T, hist_q: T):
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
    assert hist_p.get_num_bins() == hist_q.get_num_bins()

    hist_ram_p = hist_p.get_as_density()
    hist_ram_q = hist_q.get_as_density()

    bin_area = hist_p.get_bin_area()

    total_variation_ram = (
        0.5 * np.sum(np.abs(hist_ram_p - hist_ram_q)) * bin_area
    )  # To get the properly normalized integral / KLD
    return float(total_variation_ram)


get_histogram_total_variation_distance.id = "tv"
