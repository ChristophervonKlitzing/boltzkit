import numpy as np

from boltzkit.utils.histogram import Histogram1D, get_histogram_1d


def get_bond_length_hist(
    bond_lengths: np.ndarray, max_bond_length: float | None = None, **kwargs
):
    if max_bond_length is None:
        max_bond_length = bond_lengths.max()

    data_range = (0, max_bond_length)
    return get_histogram_1d(bond_lengths, data_range=data_range, **kwargs)


def get_bond_angle_hist(bond_angles: np.ndarray, **kwargs):
    data_range = (0, np.pi)
    return get_histogram_1d(bond_angles, data_range=data_range, **kwargs)


def get_dihedral_angle_hist(dihedral_angles: np.ndarray, **kwargs):
    data_range = (-np.pi, np.pi)
    return get_histogram_1d(dihedral_angles, data_range=data_range, **kwargs)
