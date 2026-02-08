import numpy as np
import openmm as mm


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
