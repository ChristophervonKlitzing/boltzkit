"""
Implementation of different Wasserstein metrics.
"""

import mdtraj as md
import numpy as np
import ot as pot
from openmm import app
from boltzkit.utils.molecular.marginals import get_phi_psi_vectors


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


def compute_torus_wasserstein_2(
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

    if len(ground_truth_cart.shape) == 2:
        batch = ground_truth_cart.shape[0]
        ground_truth_cart = ground_truth_cart.reshape((batch, -1, 3))
        model_samples_cart = model_samples_cart.reshape((batch, -1, 3))

    if isinstance(topology, app.Topology):
        topology = md.Topology.from_openmm(topology)

    phis_true, psis_true = get_phi_psi_vectors(ground_truth_cart, topology)
    angles_true = np.concatenate([phis_true, psis_true], axis=1)

    phis_pred, psis_pred = get_phi_psi_vectors(model_samples_cart, topology)
    angles_pred = np.concatenate([phis_pred, psis_pred], axis=1)

    t_wasserstein_2 = _torus_wasserstein(angles_true, angles_pred)
    return t_wasserstein_2


def compute_euclidean_wasserstein_1_2(
    X1: np.ndarray,
    X2: np.ndarray,
    weights1: np.ndarray | None = None,
    num_iter_max: int = int(1e9),
    include_w1: bool = True,
    include_w2: bool = True,
):
    """
    Compute empirical Wasserstein-1 (W1) and Wasserstein-2 (W2) distances
    between two point clouds using exact optimal transport.

    Parameters
    ----------
    X1 : np.ndarray of shape (N, d)
        First point cloud with N samples in d dimensions.
    X2 : np.ndarray of shape (M, d)
        Second point cloud with M samples in d dimensions.
    weights1 : np.ndarray of shape (N,), optional
        Non-negative weights for samples in `X1`. If None, uniform
        weights are used. The second point cloud always uses uniform weights.
    num_iter_max : int, default=1e6
        Maximum number of iterations for the optimal transport solver.
    include_w1 : bool, default=True
        If True, compute and return Wasserstein-1 distance.
    include_w2 : bool, default=True
        If True, compute and return Wasserstein-2 distance.

    Returns
    -------
    W1 : float or None
        Wasserstein-1 distance using Euclidean ground cost.
        Returns None if `include_w1=False`.
    W2 : float or None
        Wasserstein-2 distance using squared Euclidean ground cost.
        Returns None if `include_w2=False`.

    Notes
    -----
    - Exact optimal transport is computed using `ot.emd2`.
    - W1 is computed as:

          W1 = emd2(a, b, ||x - y||)

    - W2 is computed as:

          W2 = sqrt(emd2(a, b, ||x - y||^2))

    - Computational complexity is O(NM) in memory and
      typically super-cubic in time for exact OT.
    """
    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError("X1 and X2 must be 2D arrays of shape (n_samples, dim).")
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("Point clouds must have the same dimensionality.")

    N, M = X1.shape[0], X2.shape[0]

    # probability weights
    if weights1 is None:
        a = np.ones(N) / N
    else:
        a = weights1.flatten()
        a = a / np.sum(a)

    b = np.ones(M) / M

    # cost matrix: pairwise Euclidean distances
    Mmat = pot.dist(X1, X2, metric="euclidean")  # shape (N, M)

    if include_w2:
        # ---- Wasserstein-2 ----
        # cost = ||x - y||^2
        W2 = np.sqrt(pot.emd2(a, b, Mmat**2, numItermax=num_iter_max))
        W2 = float(W2.item())
    else:
        W2 = None

    if include_w1:
        # ---- Wasserstein-1 ----
        # cost = ||x - y||
        W1 = pot.emd2(a, b, Mmat, numItermax=num_iter_max)
        W1 = float(W1)
    else:
        W1 = None

    return W1, W2


if __name__ == "__main__":
    from boltzkit.targets.boltzmann import MolecularBoltzmann

    bm = MolecularBoltzmann("datasets/chrklitz99/test_system")
    topology = bm.get_openmm_topology()

    gt_samples = np.random.randn(1_000, 66)
    model_samples = np.random.randn(1_000, 66)
    T_W2 = compute_torus_wasserstein_2(gt_samples, model_samples, topology)
    print(T_W2)

    W1, W2 = compute_euclidean_wasserstein_1_2(model_samples, gt_samples)
    print(W1, W2)
