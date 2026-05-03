import numpy as np

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from boltzkit.utils.cached_repo import CachedRepo


class Dataset:
    """
    Dataset container for samples drawn from a Boltzmann distribution.

    The class stores samples together with thermodynamic quantities such as
    energies, log-probabilities, scores, or forces. Missing quantities are
    computed lazily from the available ones using the thermodynamic relations
    between energy, probability, and forces.

    The following relationships are assumed:

        log p(x) = -E(x) / (k_B T)
        score(x) = ∇_x log p(x)
        force(x) = -∇_x E(x)

    which implies

        score(x) = force(x) / (k_B T)

    Internally, all vector quantities are represented in flattened form
    with shape (batch, d). Inputs with atomic coordinate shape
    (batch, n_atoms, 3) are automatically reshaped to this representation.

    The class performs consistency checks on batch dimensions and ensures
    that incompatible quantities (e.g., energies and log-probabilities, or
    scores and forces) are not provided simultaneously.
    """

    def __init__(
        self,
        kB_T: float,
        *,
        samples: np.ndarray | None = None,
        log_probs: np.ndarray | None = None,
        energies: np.ndarray | None = None,
        scores: np.ndarray | None = None,
        forces: np.ndarray | None = None,
    ):
        """
        This class stores samples together with associated energies, log-probabilities,
        scores, or forces. Related quantities are computed lazily when accessed.

        The following physical relationships are assumed:

        - log p(x) = -E(x) / (k_B T)
        - score(x) = ∇_x log p(x)
        - force(x) = -∇_x E(x)

        which implies

        - score(x) = force(x) / (k_B T)

        Either energies or log-probabilities may be provided, but not both.
        Likewise, either scores or forces may be provided.

        Parameters
        ----------
        kB_T : float
            Thermal energy (Boltzmann constant times temperature). Must be positive.

        samples : ndarray, optional
            Sample configurations with shape (batch, d) or (batch, #atoms, 3).

        log_probs : ndarray, optional
            Log-probabilities of samples with shape (batch,) or (batch, 1).

        energies : ndarray, optional
            Energies of samples with shape (batch,) or (batch, 1).

        scores : ndarray, optional
            Score function values ∇_x log p(x) with shape (batch, d) or
            (batch, n_atoms, 3).

        forces : ndarray, optional
            Forces -∇_x E(x) with shape (batch, d) or (batch, n_atoms, 3).

        Notes
        -----
        - All provided arrays must share the same first dimension (batch size).
        - Arrays with shape (batch, n_atoms, 3) are flattened internally to
          shape (batch, n_atoms * 3).
        - Arrays with shape (batch, 1) are converted to shape (batch,).
        - Missing quantities are computed on demand from the available ones.
        """

        if log_probs is not None and energies is not None:
            raise ValueError("Only one of log_probs or energies can be provided")

        if scores is not None and forces is not None:
            raise ValueError("Only one of scores or forces can be provided")

        if kB_T <= 0:
            raise ValueError("kB_T must be positive")

        self._dataset_size = self._check_same_batch_size(
            samples=samples,
            log_probs=log_probs,
            energies=energies,
            scores=scores,
            forces=forces,
        )

        samples = self._cast_2d(samples)
        log_probs = self._cast_1d(log_probs)
        energies = self._cast_1d(energies)
        scores = self._cast_2d(scores)
        forces = self._cast_2d(forces)

        # log_prob = -energy / kB_T
        # score = forces / kB_T

        self._kB_T = kB_T

        self._samples = samples
        self._log_probs = log_probs
        self._energies = energies
        self._scores = scores
        self._forces = forces

    def _check_same_batch_size(self, **kwargs: np.ndarray):
        batch_size = None
        for a in kwargs.values():
            if a is not None:
                batch_size = a.shape[0]
                break

        if batch_size is None:
            raise ValueError(
                "At least one array must be provided (all inputs were None)."
            )

        for key, val in kwargs.items():
            if val is not None and val.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: expected first dimension {batch_size}, "
                    f"but '{key}' has shape {val.shape}."
                )
        return batch_size

    def _cast_1d(self, val: np.ndarray):
        if val is None:
            return None

        if val.ndim > 2 or val.ndim == 2 and val.shape[1] != 1:
            raise ValueError(
                f"Expected a 1D array or a 2D column array with shape (batch,) or (batch, 1), "
                f"but got array with shape {val.shape}."
            )

        return val.reshape((val.shape[0],))

    def _cast_2d(self, val: np.ndarray):
        if val is None:
            return None

        if val.ndim == 2:
            return val

        if val.ndim == 3:
            if val.shape[2] != 3:
                raise ValueError(
                    f"Expected shape (batch, n_atoms, 3), but got {val.shape}."
                )
            batch = val.shape[0]
            return val.reshape((batch, -1))

        raise ValueError(
            f"Expected array with shape (batch, d) or (batch, n_atoms, 3), "
            f"but got shape {val.shape}."
        )

    @property
    def size(self):
        return self._dataset_size

    def __len__(self):
        return self._dataset_size

    def __get_sized_array(self, array: np.ndarray | None, length: int, array_type: str):
        if not (length == -1 or length > 0):
            raise ValueError(
                f"The specified length must satisfy `length == -1 or length > 0` but got {length}"
            )
        if length > self.size:
            raise ValueError(
                f"The requested number of {array_type} ({length}) is larger than the dataset size ({self.size})"
            )

        if array is not None and length != -1:
            return array[:length]
        else:
            return array

    def get_samples(self, length: int = -1) -> np.ndarray | None:
        """
        Sample configurations.

        Returns
        -------
        ndarray or None
            Array with shape (batch, d), or ``None`` if no samples were provided.
        """
        return self.__get_sized_array(self._samples, length, "samples")

    def get_log_probs(self, length: int = -1) -> np.ndarray | None:
        """
        Log-probabilities of the samples.

        If log-probabilities were not provided during initialization but
        energies were given, they are computed as

            log p(x) = -E(x) / (k_B T)

        Returns
        -------
        ndarray or None
            Array with shape (batch,), or ``None`` if neither energies nor
            log-probabilities are available.
        """
        log_probs = None

        if self._log_probs is not None:
            log_probs = self._log_probs

        if self._energies is not None:
            log_probs = -self._energies / self._kB_T

        return self.__get_sized_array(log_probs, length, "log_probs")

    def get_energies(self, length: int = -1) -> np.ndarray | None:
        """
        Energies of the samples.

        If energies were not provided during initialization but
        log-probabilities were given, they are computed as

            E(x) = -log p(x) * (k_B T)

        Returns
        -------
        ndarray or None
            Array with shape (batch,), or ``None`` if neither energies nor
            log-probabilities are available.
        """
        energies = None

        if self._energies is not None:
            energies = self._energies

        if self._log_probs is not None:
            energies = -self._log_probs * self._kB_T

        return self.__get_sized_array(energies, length, "energies")

    def get_scores(self, length: int = -1) -> np.ndarray | None:
        """
        Score function values ∇_x log p(x).

        If scores were not provided but forces were given, they are computed as

            score(x) = force(x) / (k_B T)

        Returns
        -------
        ndarray or None
            Array with shape (batch, d), or ``None`` if neither scores nor
            forces are available.
        """
        scores = None

        if self._scores is not None:
            scores = self._scores

        if self._forces is not None:
            scores = self._forces / self._kB_T

        return self.__get_sized_array(scores, length, "scores")

    def get_forces(self, length: int = -1) -> np.ndarray | None:
        """
        Forces acting on the samples.

        Forces are defined as

            force(x) = -∇_x E(x)

        If forces were not provided but scores were given, they are computed as

            force(x) = score(x) * (k_B T)

        Returns
        -------
        ndarray or None
            Array with shape (batch, d), or ``None`` if neither forces nor
            scores are available.
        """
        forces = None

        if self._forces is not None:
            forces = self._forces

        if self._scores is not None:
            forces = self._scores * self._kB_T

        return self.__get_sized_array(forces, length, "forces")

    @classmethod
    def create_from_cached_repo(
        cls,
        repo: "CachedRepo",
        type,
        length,
        kB_T: float,
        include_samples: bool,
        include_log_probs: bool,
        include_scores: bool,
        allow_autogen: bool,
        cache_log_probs: bool,
        cache_scores: bool,
        log_prob_fn: Callable[[np.ndarray], np.ndarray] | None,
        score_fn: Callable[[np.ndarray], np.ndarray] | None,
    ):
        from boltzkit.utils.dataset_helper import create_dataset_from_cached_repo

        return create_dataset_from_cached_repo(
            repo=repo,
            type=type,
            length=length,
            kB_T=kB_T,
            include_samples=include_samples,
            include_log_probs=include_log_probs,
            include_scores=include_scores,
            allow_autogen=allow_autogen,
            cache_log_probs=cache_log_probs,
            cache_scores=cache_scores,
            log_prob_fn=log_prob_fn,
            score_fn=score_fn,
        )
