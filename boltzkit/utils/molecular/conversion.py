import openmm as mm
import numpy as np


def numpy_to_vec3_list(pos: np.ndarray) -> list[mm.Vec3]:
    """
    Convert numpy position vector in nanometers into list openmm Vec3 entries.

    :param pos: Position array of shape (n, 3)
    :type pos: np.ndarray
    :return: Single molecule atom positions
    :rtype: list[mm.Vec3]
    """
    return [mm.Vec3(*pos[i].tolist()) for i in range(pos.shape[0])]


def vec3_list_to_numpy(pos: list[mm.Vec3]) -> np.ndarray:
    return np.asarray([[p.x, p.y, p.z] for p in pos])


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
