"""
Implementation of different Wasserstein metrics.
"""

import numpy as np
import ot as pot


def get_torus_wasserstein(angles0: np.ndarray, angles1: np.ndarray) -> float:
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


def get_euclidean_wasserstein_1_2(
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
    from boltzkit.evaluation.sample_based.torsion_marginals import get_torsion_angles

    rng = np.random.default_rng(0)

    bm = MolecularBoltzmann("datasets/chrklitz99/alanine_dipeptide")
    topology = bm.get_mdtraj_topology()
    dataset = bm.load_dataset(300, "val", length=20_000)
    val_samples = dataset.get_samples()

    angles = get_torsion_angles(val_samples, topology)
    angles = np.concatenate(angles, axis=1)
    print(angles.shape)

    n_samples = 5_000

    distances_true = []
    for _ in range(10):
        perm = rng.permutation(val_samples.shape[0])
        mask_a = perm[:n_samples]
        mask_b = perm[n_samples : 2 * n_samples]

        print(mask_a.shape, mask_b.shape)
        # Select random subset of torsion angles
        angles_a = angles[mask_a]
        angles_b = angles[mask_b]

        W2 = get_torus_wasserstein(angles_a, angles_b)
        print(W2)
        distances_true.append(W2)

    angles_pred = get_torsion_angles(
        val_samples + rng.normal(size=val_samples.shape) * 0.01, topology
    )
    angles_pred = np.concatenate(angles_pred, axis=1)
    print(angles_pred.shape)

    n_samples = 5_000

    distances_pred = []
    for _ in range(10):
        perm = rng.permutation(val_samples.shape[0])
        mask_a = perm[:n_samples]
        mask_b = perm[n_samples : 2 * n_samples]

        print(mask_a.shape, mask_b.shape)
        # Select random subset of torsion angles
        angles_a = angles_pred[mask_a]
        angles_b = angles[mask_b]

        W2 = get_torus_wasserstein(angles_a, angles_b)
        print(W2)
        distances_pred.append(W2)

    print(f"W2_internal: {np.mean(distances_true)}±{np.std(distances_true)}")
    print(f"W2_cross: {np.mean(distances_pred)}±{np.std(distances_pred)}")

    import numpy as np
    from scipy import stats

    A = np.array(distances_true)
    B = np.array(distances_pred)

    t_stat, p_value = stats.ttest_rel(A, B)

    alpha = 0.05
    if p_value < alpha:
        print("Significant difference (p =", p_value, ")")
    else:
        print("Not significant (p =", p_value, ")")
