import deeptime as dt
import mdtraj as md
import numpy as np

SELECTION = "symbol == C or symbol == N or symbol == S"


def _wrap_angle(array):
    return (np.sin(array), np.cos(array))


def _get_distances(xyz):
    distance_matrix_ca: np.ndarray = np.linalg.norm(
        xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1
    )
    n_ca = distance_matrix_ca.shape[-1]
    m, n = np.triu_indices(n_ca, k=1)
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca


def get_tica_features(
    trajectory: md.Trajectory,
    use_dihedrals: bool = True,
    use_distances: bool = True,
    selection: str = SELECTION,
):
    top: md.Topology = trajectory.topology
    trajectory = trajectory.atom_slice(top.select(selection))
    features: list[np.ndarray] = []  # list of arrays of shape (traj_length, d_i)

    if use_dihedrals:
        _, phi = md.compute_phi(trajectory)
        _, psi = md.compute_psi(trajectory)
        _, omega = md.compute_omega(trajectory)
        dihedrals = np.concatenate(
            [*_wrap_angle(phi), *_wrap_angle(psi), *_wrap_angle(omega)], axis=-1
        )
        features.append(dihedrals)

    if use_distances:
        ca_distances = _get_distances(trajectory.xyz)
        features.append(ca_distances)

    if len(features) == 0:
        raise ValueError(
            "No TICA features selected. Enable at least one feature type "
            "(use_dihedrals or use_distances)."
        )

    return np.concatenate(features, -1)


def create_tica_model(
    trajectory: md.Trajectory,
    lagtime: int,
    dim: int = 40,
    use_dihedrals=True,
    use_distances=True,
    use_koopman=True,
):
    ca_features = get_tica_features(
        trajectory, use_dihedrals=use_dihedrals, use_distances=use_distances
    )
    tica = dt.decomposition.TICA(dim=dim, lagtime=lagtime)

    if use_koopman:
        koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
        reweighting_model = koopman_estimator.fit(ca_features).fetch_model()
        tica_model = tica.fit(ca_features, reweighting_model).fetch_model()
    else:
        tica_model = tica.fit(ca_features).fetch_model()

    return tica_model
