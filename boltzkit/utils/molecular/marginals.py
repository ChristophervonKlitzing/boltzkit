import mdtraj as md
import numpy as np


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
    traj_samples = md.Trajectory(samples, topology=topology)
    phis = md.compute_phi(traj_samples)[1]
    psis = md.compute_psi(traj_samples)[1]
    return phis, psis
