import pickle
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
    """
    Compute feature vectors for TICA from an MDTraj trajectory.

    Features can include pairwise distances between selected atoms and
    backbone dihedral angles represented using sine/cosine encoding.

    Parameters
    ----------
    trajectory : md.Trajectory
        Input trajectory containing atomic coordinates and topology.
    use_dihedrals : bool, default=True
        Whether to include backbone dihedral angles (phi, psi, omega)
        as features.
    use_distances : bool, default=True
        Whether to include pairwise distances between selected atoms.
    selection : str, default=SELECTION
        MDTraj atom selection string used to filter atoms before
        computing features.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_frames, n_features).

    Raises
    ------
    ValueError
        If both `use_dihedrals` and `use_distances` are False.

    Notes
    -----
    - Distances are computed for all unique atom pairs within the
      selected atom subset.
    - Dihedral angles are transformed using sine/cosine encoding to
      avoid discontinuities due to angular periodicity.
    """
    top: md.Topology = trajectory.topology
    trajectory = trajectory.atom_slice(top.select(selection))
    features: list[np.ndarray] = []  # list of arrays of shape (traj_length, d_i)

    if use_distances:
        ca_distances = _get_distances(trajectory.xyz)
        features.append(ca_distances)

    if use_dihedrals:
        _, phi = md.compute_phi(trajectory)
        _, psi = md.compute_phi(trajectory)
        _, omega = md.compute_omega(trajectory)
        dihedrals = np.concatenate(
            [*_wrap_angle(phi), *_wrap_angle(psi), *_wrap_angle(omega)], axis=-1
        )
        features.append(dihedrals)

    if len(features) == 0:
        raise ValueError(
            "No TICA features selected. Enable at least one feature type "
            "(use_dihedrals or use_distances)."
        )

    return np.concatenate(features, -1)


def create_deeptime_tica_model(
    trajectory: md.Trajectory,
    lagtime: int,
    dim: int = 40,
    use_dihedrals=True,
    use_distances=True,
    use_koopman=True,
):
    """
    Train a TICA model using the deeptime library.

    Parameters
    ----------
    trajectory : md.Trajectory
        Input molecular dynamics trajectory.
    lagtime : int
        Lag time (in frames) used for TICA estimation.
    dim : int, default=40
        Number of TICA components to retain.
    use_dihedrals : bool, default=True
        Whether to include dihedral-angle features.
    use_distances : bool, default=True
        Whether to include pairwise distance features.
    use_koopman : bool, default=True
        If True, apply Koopman reweighting to correct for sampling bias.

    Returns
    -------
    dt.decomposition.TransferOperatorModel
        Fitted TICA model from the deeptime library.

    Notes
    -----
    Koopman reweighting can improve estimates when trajectories are
    not sampled from the equilibrium distribution.
    """
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


class TicaModelWithLengthScale:
    """
    Wrapper around a deeptime TICA model that applies a coordinate
    length-scale conversion before feature extraction.

    This is useful when the TICA model was trained using coordinates in
    one unit system (e.g., nanometers) but new data is provided in a
    different unit system (e.g., angstroms).
    """

    def __init__(
        self,
        model: dt.decomposition.TransferOperatorModel,
        length_scale: float = 1.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : dt.decomposition.TransferOperatorModel
            A fitted deeptime transfer operator model (e.g., a TICA model).
        length_scale : float, default=1.0
            Multiplicative factor applied to Cartesian coordinates before
            computing TICA features.

            Example:
            - Training data in nanometers
            - Inference data in angstroms
            → use `length_scale=0.1`.

        Raises
        ------
        ValueError
            If the provided model is not a deeptime TransferOperatorModel.
        """

        super().__init__()
        if not isinstance(model, dt.decomposition.TransferOperatorModel):
            raise ValueError(
                f"Input of type '{type(model).__name__}' currently not supported"
            )
        self._model = model
        self._length_scale = length_scale
        self._tica_feature_args = kwargs

    def project_from_cartesian(
        self, pos: np.ndarray, topology: md.Topology
    ) -> np.ndarray:
        """
        Project Cartesian coordinates onto the TICA components.

        Parameters
        ----------
        pos : np.ndarray
            Flattened Cartesian coordinates of shape (n_frames, n_atoms * 3)
            or compatible with reshaping to (n_frames, n_atoms, 3).
        topology : md.Topology
            MDTraj topology corresponding to the coordinates.

        Returns
        -------
        np.ndarray
            TICA projections of shape (n_frames, n_tica_components).

        Notes
        -----
        - Coordinates are first rescaled using the model's `length_scale`.
        - Feature extraction is performed using :func:`get_tica_features`.
        - Data is processed in batches of 4096 frames to limit memory usage.
        """
        pos = pos.reshape(pos.shape[0], -1, 3) * self._length_scale
        ticas_list = []
        for i in range(0, pos.shape[0], 4096):
            tica_features_batch = get_tica_features(
                md.Trajectory(
                    xyz=pos[i : i + 4096],
                    topology=topology,
                ),
                **self._tica_feature_args,
            )
            tica_projections_batch = self._model.transform(tica_features_batch)
            ticas_list.append(tica_projections_batch)
        return np.concatenate(ticas_list, axis=0)

    @classmethod
    def from_path(cls, path: str, length_scale: float):
        """
        Load a serialized TICA model and wrap it with a length-scale
        conversion.

        Parameters
        ----------
        path : str
            Path to a pickled deeptime TICA model.
        length_scale : float
            Coordinate scaling factor applied during projection.

        Returns
        -------
        TicaModelWithLengthScale
            Wrapper instance containing the loaded TICA model.
        """
        with open(path, "rb") as f:
            tica_model: dt.decomposition.TICA = pickle.load(f)
        return TicaModelWithLengthScale(tica_model, length_scale=length_scale)
