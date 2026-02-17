"""
Implementation of different Wasserstein metrics.
"""

import mdtraj as md
import numpy as np
import ot as pot
from openmm import app


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
    samples = samples
    traj_samples = md.Trajectory(samples, topology=topology)
    phis = md.compute_phi(traj_samples)[1]
    psis = md.compute_psi(traj_samples)[1]
    return phis, psis


def _torus_wasserstein(angles0: np.ndarray, angles1: np.ndarray) -> float:
    """
    Compute the 2-Wasserstein distance between two sets of angular samples
    on a torus (periodic domain).

    Parameters
    ----------
    angles0 : np.ndarray
        First set of angular samples with shape (n_samples_0, d),
        where d is the number of angular dimensions.
    angles1 : np.ndarray
        Second set of angular samples with shape (n_samples_1, d).

    Returns
    -------
    float
        The 2-Wasserstein distance between the empirical distributions
        defined by `angles0` and `angles1`.

    Notes
    -----
    - Distances are computed using the wrapped (circular) metric:
      min(|Δθ|, 2π - |Δθ|).
    - The squared Euclidean distance on the torus is used as the ground cost.
    - Uniform weights are assumed for both empirical distributions.
    - The optimal transport problem is solved using POT's `emd2`.
    """

    # weights:
    a, b = pot.unif(angles0.shape[0]), pot.unif(angles1.shape[0])

    # wrapped (circular) distances:
    angles0 = angles0[:, None]
    angles1 = angles1[None, :]
    dists = np.minimum(np.abs(angles0 - angles1), 2 * np.pi - np.abs(angles0 - angles1))
    dists = dists**2

    # Compute Wasserstein distance using POT
    dist_squared = pot.emd2(a, b, dists.sum(-1), numItermax=int(1e9))
    return float(np.sqrt(dist_squared).item())


def compute_torus_wasserstein(
    ground_truth_cart: np.ndarray,
    model_samples_cart: np.ndarray,
    topology: md.Topology | app.Topology,
) -> float:
    """
    Compute the toroidal 2-Wasserstein distance between backbone torsion
    angle distributions of two molecular ensembles.

    Parameters
    ----------
    ground_truth_cart : np.ndarray
        Cartesian coordinates of the reference ensemble with shape
        (batch, n_atoms, 3) or (batch, n_atoms * 3)
    model_samples_cart : np.ndarray
        Cartesian coordinates of the model-generated ensemble with shape
        (batch, n_atoms, 3) or (batch, n_atoms * 3)
    topology : md.Topology or openmm.app.Topology
        Molecular topology corresponding to the atomic coordinates.
        If an OpenMM topology is provided, it will be converted to MDTraj format.

    Returns
    -------
    float
        The toroidal 2-Wasserstein distance between the joint φ/ψ
        angle distributions of the two ensembles.

    Notes
    -----
    - Backbone φ and ψ torsion angles are extracted for each element in the batch.
    - The joint angular distribution is treated as living on a torus.
    - Uniform weights are assumed for both ensembles.
    """
    batch = ground_truth_cart.shape[0]
    n_atoms = topology.getNumAtoms()

    ground_truth_cart = ground_truth_cart.reshape((batch, n_atoms, -1))
    model_samples_cart = model_samples_cart.reshape((batch, n_atoms, -1))

    if isinstance(topology, app.Topology):
        topology = md.Topology.from_openmm(topology)

    phis_true, psis_true = get_phi_psi_vectors(ground_truth_cart, topology)
    angles_true = np.concatenate([phis_true, psis_true], axis=1)

    phis_pred, psis_pred = get_phi_psi_vectors(model_samples_cart, topology)
    angles_pred = np.concatenate([phis_pred, psis_pred], axis=1)

    t_wasserstein_2 = _torus_wasserstein(angles_true, angles_pred)
    return t_wasserstein_2


if __name__ == "__main__":
    from boltzkit.targets.boltzmann import MolecularBoltzmann

    bm = MolecularBoltzmann("datasets/chrklitz99/test_system")
    topology = bm.get_openmm_topology()

    gt_samples = np.random.randn(1_000, 66)
    model_samples = np.random.randn(1_000, 66)
    T_W2 = compute_torus_wasserstein(gt_samples, model_samples, topology)
    print(T_W2)
