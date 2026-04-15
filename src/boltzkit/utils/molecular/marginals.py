import mdtraj as md
import numpy as np


def get_trajectory(samples: np.ndarray, topology: md.Topology):
    batch = samples.shape[0]
    samples = samples.reshape(batch, -1, 3)
    assert topology.n_atoms == samples.shape[1]

    traj_samples = md.Trajectory(samples, topology=topology)
    return traj_samples


def get_phi_psi_vectors(samples: np.ndarray, topology: md.Topology):
    """
    Compute backbone φ (phi) and ψ (psi) torsion angles from Cartesian samples.

    Parameters
    ----------
    samples : np.ndarray
        Cartesian coordinates with shape (batch, n_atoms, 3).
    topology : md.Topology
        MDTraj topology corresponding to the atomic coordinates.

    Returns
    -------
    phis : np.ndarray
        Array of phi torsion angles in radians with shape (batch, n_phi).
    psis : np.ndarray
        Array of psi torsion angles in radians with shape (batch, n_psi).

    Notes
    -----
    The angles are computed using MDTraj's `compute_phi` and `compute_psi`
    functions. Returned angles are in radians and lie in the interval [-π, π].
    """

    traj_samples = get_trajectory(samples, topology)
    phis = md.compute_phi(traj_samples)[1]
    psis = md.compute_psi(traj_samples)[1]
    return phis, psis


def filter_z_matrix_columns(
    z_matrix: list[tuple[int, int, int, int]],
    num_columns: int,
    start: int = 0,
    filter_None=True,
    filter_negative=True,
):
    assert num_columns > 0
    end = start + num_columns

    def is_valid(row: tuple[int, ...]) -> bool:
        valid = True
        if filter_None:
            valid = valid and all([r is not None for r in row])
        if valid and filter_negative:
            valid = valid and all([r >= 0 for r in row])
        return valid

    return [row[start:end] for row in z_matrix if is_valid(row[start:end])]


def get_bond_lengths(
    samples: np.ndarray,
    topology: md.Topology,
    z_matrix: list[tuple[int, int, int, int]],
) -> np.ndarray:
    # bond-lengths >= 0, returns (batch, n_bond_lengths)
    atom_pairs = filter_z_matrix_columns(z_matrix, 2)
    traj_samples = get_trajectory(samples, topology)
    return md.compute_distances(traj_samples, atom_pairs=atom_pairs)


def get_bond_angles(
    samples: np.ndarray,
    topology: md.Topology,
    z_matrix: list[tuple[int, int, int, int]],
) -> np.ndarray:
    # bond angles in radians in range [0, pi], returns (batch, n_bond_angles)
    bond_angle_triplets = filter_z_matrix_columns(z_matrix, 3)
    traj_samples = get_trajectory(samples, topology)
    return md.compute_angles(traj_samples, angle_indices=bond_angle_triplets)


def get_dihedral_angles(
    samples: np.ndarray,
    topology: md.Topology,
    z_matrix: list[tuple[int, int, int, int]],
) -> np.ndarray:
    # dihedral angles in radians in range (-pi, pi], returns (batch, n_dihedral_angles)
    dihedral_angle_quartets = filter_z_matrix_columns(z_matrix, 4)
    traj_samples = get_trajectory(samples, topology)
    return md.compute_dihedrals(traj_samples, indices=dihedral_angle_quartets)
