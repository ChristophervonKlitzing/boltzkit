from abc import ABC
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pickle
from typing import Callable, Literal, TYPE_CHECKING, TypedDict

import numpy as np
import openmm.app as app
import mdtraj as md

from boltzkit.utils.dataset import Dataset

if TYPE_CHECKING:
    from boltzkit.utils.cached_repo import CachedRepo


@lru_cache
def load_from_file(
    path: str | Path,
    data_type: Literal["log_probs", "samples"],
    n_samples: int | None = None,
    dtype: type | np.dtype = np.float32,
) -> np.ndarray:
    """
    Load data from a file and validate/reshape it according to its type.

    This function supports PyTorch (`.pt`, `.pth`) and NumPy (`.npy`, `.npz`) files.
    The loaded data is converted to a NumPy array and reshaped based on `data_type`.

    Parameters
    ----------
    path : str
        Path to the file to load. Must exist and have a supported extension.
    data_type : {"log_probs", "samples"}
        Specifies the type of data being loaded, which determines shape validation:
        - "log_probs": expects data of shape (batch,) or (batch, 1) and flattens to (batch,)
        - "samples": expects data of shape (batch,), (batch, dim), or (batch, n_nodes, 3)
                     3D molecular data is flattened to (batch, n_nodes*3)
    dtype : np.dtype, optional
        Desired floating-point type for the loaded data. The data will be converted
        to this type after loading. If not specified, the library's default
        floating-point type (`np.float64`) is used.

    Returns
    -------
    np.ndarray
        Loaded data as a NumPy array with appropriate shape for the given `data_type`.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ImportError
        If the file format requires PyTorch or NumPy and the library is not installed.
    RuntimeError
        If the file could not be loaded.
    TypeError
        If the loaded object is not of the expected type (`torch.Tensor` for PyTorch, `np.ndarray` for NumPy).
    ValueError
        If the file extension is unsupported or if the loaded data has an invalid shape for the specified `data_type`.

    Examples
    --------
    >>> from pathlib import Path
    >>> data = _load_from_file(Path("predictions.npy"), data_type="log_probs")
    >>> print(data.shape)
    (1000,)

    >>> data = _load_from_file(Path("samples.pt"), data_type="samples")
    >>> print(data.shape)
    (1000, 198)  # if original shape was (1000, 66, 3) for molecular coordinates
    """

    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    suffix = path.suffix.lower()

    # -------------------------
    # PyTorch formats
    # -------------------------
    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "Loading '.pt' or '.pth' files requires PyTorch, "
                "but it could not be imported."
            ) from e

        try:
            data_torch = torch.load(path, map_location="cpu")
            if not isinstance(data_torch, torch.Tensor):
                raise TypeError(
                    f"The loaded PyTorch data from path '{path}' is not of type 'torch.Tensor'"
                )

            data = data_torch.numpy()

        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch file: {path}") from e

    # -------------------------
    # NumPy formats
    # -------------------------
    elif suffix in {".npy", ".npz"}:
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "Loading '.npy' or '.npz' files requires NumPy, "
                "but it could not be imported."
            ) from e

        try:
            data_np = np.load(path, allow_pickle=False)
            if not isinstance(data_np, np.ndarray):
                raise TypeError(
                    f"The loaded NumPy data from path '{path}' is not of type 'np.ndarray'"
                )

            data = data_np
        except Exception as e:
            raise RuntimeError(f"Failed to load NumPy file: {path}") from e

    else:
        raise ValueError(f"Unsupported file format '{suffix}' for file: {path}")

    batch = data.shape[0]

    if data_type == "log_probs":
        # Ensure data is 1D: (batch,) or (batch,1)
        if (len(data.shape) == 2 and data.shape[1] != 1) or len(data.shape) > 2:
            raise ValueError(
                f"Unsupported shape for log_probs ({data.shape}). "
                "Expected (batch,) or (batch,1)."
            )
        # Flatten to (batch,)
        data = data.reshape((batch,))
    elif data_type == "samples":
        # Ensure at least 2D: (batch, dim)
        if len(data.shape) == 1:
            data = data.reshape((batch, 1))  # single feature per sample

        # Allow molecular data: (batch, n_nodes, 3)
        elif len(data.shape) == 3:
            if data.shape[-1] != 3:
                raise ValueError(
                    f"Unsupported shape for samples ({data.shape}). "
                    "Expected (batch,), (batch, dim), or (batch, n_nodes, 3)."
                )
            # Flatten node coordinates: (batch, n_nodes*3)
            data = data.reshape((batch, -1))

        # Optionally check for unsupported shapes
        elif len(data.shape) > 3:
            raise ValueError(
                f"Unsupported shape for samples ({data.shape}). "
                "Expected (batch,), (batch, dim), or (batch, n_nodes, 3)."
            )

    if n_samples is not None:
        data = data[:n_samples]

    if data.dtype != dtype:
        data = np.array(data, dtype=dtype)

    return data


@lru_cache
def load_tica_model(path: str | Path):
    # TODO: Also allow numpy and pytorch arrays and convert to deeptime tica model
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"TICA model file does not exist: {path}")

    with open(path, "rb") as f:
        tica_model = pickle.load(f)

    return tica_model


@lru_cache
def load_topology(path: str | Path):
    # TODO: Support other formats like numpy or pytorch arrays, yaml configs, pickle

    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Topology file does not exist: {path}")

    if not path.suffix.lower().endswith(".pdb"):
        raise RuntimeError(f"Could not load topology from path '{path}'")

    mm_top = app.PDBFile(path.as_posix()).topology
    return md.Topology.from_openmm(mm_top)


def cache_load_sample_derived_data(
    samples: np.ndarray,
    data_fpath: Path | None,
    data_cache_fpath: Path | None = None,
    data_eval_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    allow_autogen: bool = False,
    cache_data: bool = False,
) -> np.ndarray:
    """
    Load or compute derived data for a set of samples with optional caching.

    The function attempts, in order, to load data from a primary file, fall back
    to a cache file, or generate missing data using a provided evaluation
    function. Generated data can optionally be cached.

    Logic priority:
    1. Load from primary data_fpath if it exists.
    2. Load from data_cache_fpath if it exists (requires cache_data to be True).
    3. If allow_autogen is True, compute missing data using data_eval_fn .
    4. If cache_data is True, save computed results to data_cache_fpath.

    :param samples: Input samples of shape (n_samples, ...).
    :type samples: numpy.ndarray
    :param data_fpath: Path to the primary data file to load.
    :type data_fpath: pathlib.Path or None
    :param data_cache_fpath: Path to the cache file for loading/saving data.
    :type data_cache_fpath: pathlib.Path or None
    :param data_eval_fn: Function to compute derived data from samples (e.g., log_probs or scores).
    :type data_eval_fn: Callable[[numpy.ndarray], numpy.ndarray] or None
    :param allow_autogen: If True, compute missing data when not available.
    :type allow_autogen: bool
    :param cache_data: If True, enable loading from and saving to cache.
    :type cache_data: bool

    :returns: Array of derived data aligned with ``samples``.
    :rtype: numpy.ndarray

    :raises ValueError: If autogeneration is enabled but no evaluation function is provided.
    :raises RuntimeError: If data cannot be loaded or generated.
    """
    data: np.ndarray | None = None
    n_samples = samples.shape[0]

    # 1. Try loading from primary file path
    if data_fpath and data_fpath.exists():
        data = np.load(data_fpath)

        if data.shape[0] < n_samples:
            raise ValueError(
                f"Too few sample-derived datapoints: For {n_samples} samples, only "
                f"{data.shape[0]} sample-derived datapoints were found in '{data_fpath}'."
            )
        data = data[:n_samples]

    if data is None and cache_data and not allow_autogen:
        raise ValueError("Cannot use caching without `auto_gen=True`")

    # 2. Try loading from cache if primary file wasn't found/provided
    if data is None and cache_data and data_cache_fpath and data_cache_fpath.exists():
        data = np.load(data_cache_fpath)

    # Trim data if it's longer than current samples
    if data is not None and data.shape[0] > n_samples:
        data = data[:n_samples]

    # 3. Handle Autogeneration
    if allow_autogen:
        n_existing = 0 if data is None else data.shape[0]

        if n_existing < n_samples:
            if data_eval_fn is None:
                raise ValueError("Autogen enabled but data_eval_fn is None.")

            # Compute only the missing part
            missing_data = data_eval_fn(samples[n_existing:])

            if data is None:
                data = missing_data
            else:
                data = np.concatenate([data, missing_data], axis=0)

            # 4. Handle Caching of newly generated data
            if cache_data and data_cache_fpath is not None:
                data_cache_fpath.parent.mkdir(parents=True, exist_ok=True)
                np.save(data_cache_fpath, data)

    if data is None:
        raise RuntimeError("Data could not be loaded from file, cache, or autogen.")

    return data


class DatasetLoader(ABC):
    def load_dataset(
        self,
        type: Literal["train", "val", "test"],
        length: int,
        *,
        include_samples: bool = True,
        include_log_probs: bool = False,
        include_scores: bool = False,
        **kwargs,
    ) -> Dataset:
        """
        Load the dataset of the specified split.

        This method retrieves samples and optionally associated
        log_probs/energies and scores/forces.

        Parameters
        ----------
        type : Literal["train", "val", "test"]
            Dataset split to load.
        length : int, optional
            Maximum number of samples to load. If -1, all available samples are used.
        T : float | int | None
            Temperature (in Kelvin) identifying the dataset. Integers are cast to float. If None, the target's temperature is used.
        include_samples : bool, default=True
            Whether to return samples.
        include_log_probs : bool, default=False
            Whether to include energy values for each sample. Fails if no energies are available and `allow_autogen` is False.
        include_scores : bool, default=False
            Whether to include force values for each sample. Fails if no forces are available and `allow_autogen` is False.

        Returns
        -------
        Dataset

        Raises
        ------
        ValueError | NotImplementedError | Exception
            If dataset configuration is missing or cannot be computed/retrieved
        """

        raise NotImplementedError

    def try_load_dataset(self, *args, **kwargs) -> Dataset | str:
        """
        Same input as `load_dataset` but instead of failing on a missing dataset, the error message is returned.
        """
        try:
            dataset_or_error_msg = self.load_dataset(*args, **kwargs)
        except Exception as e:
            dataset_or_error_msg = str(e)

        return dataset_or_error_msg


def _get_dataset_config_from_cached_repo(repo: "CachedRepo", type: str, T: float = 1.0):
    datasets: dict[str, dict[str, str]] | None = repo.config.get("datasets", None)
    if datasets is None:
        raise ValueError("Missing datasets config")

    temp_cfg = datasets.get(str(T), None)
    if temp_cfg is None:
        available_temps = list(datasets.keys())
        raise RuntimeError(
            f"Missing dataset: "
            f"Searched for temperature {T}K, but only found {available_temps}."
        )

    dset_cfg: dict[str, str] | str | None = temp_cfg.get(type, None)
    if dset_cfg is None:
        available_keys = list(temp_cfg.keys())
        raise RuntimeError(
            f"Missing dataset type for temperature {T}K. "
            f"Searched for type '{type}', but only found {available_keys}."
        )

    if isinstance(dset_cfg, str):
        dset_cfg = {"samples": dset_cfg}

    return dset_cfg


def _get_cache_path(
    samples_fpath: Path,
    cache_data_type: Literal["log_probs", "scores"] | str,
) -> Path:
    """
    Creates a cache path next to the samples file, e.g.,
    'samples.npy' -> 'samples.npy_log_probs.npy'
    """
    return samples_fpath.with_name(f"{samples_fpath.name}_cache_{cache_data_type}.npy")


@dataclass(frozen=True)
class CacheLoadingArgs:
    """
    Configuration options controlling dataset cache loading and automatic
    generation of sample-related quantities (log-probs/energies, scores/forces).

    Parameters
    ----------
    allow_autogen : bool, optional, default=True
        If ``True``, missing quantities (e.g., log-probs/energies, scores/forces)
        may be computed automatically online if possible.

    cache_log_probs : bool, optional, default=True
        Whether log-probs/energies can be cached after online-computation (allow_autogen=True)
        or loaded from cache files if available.

    cache_scores : bool, optional, default=False
        Whether scores/forces can be cached after online-computation (allow_autogen=True)
        or loaded from cache files if available.
    """

    allow_autogen: bool = True
    cache_log_probs: bool = True
    cache_scores: bool = False


class CachedRepoDatasetLoader(DatasetLoader):
    def __init__(
        self,
        kB_T: float,
        cached_repo: "CachedRepo",
        T: float,
        log_prob_fn: Callable[[np.ndarray], np.ndarray] | None,
        score_fn: Callable[[np.ndarray], np.ndarray] | None,
        caching_args: CacheLoadingArgs | dict | None = None,
    ):
        super().__init__()

        if caching_args is None:
            caching_args = CacheLoadingArgs()
        elif isinstance(caching_args, dict):
            caching_args = CacheLoadingArgs(**caching_args)

        self.__kB_T = kB_T
        self.__repo = cached_repo
        self._T = T

        self.__log_prob_fn = log_prob_fn
        self.__score_fn = score_fn

        self.__allow_autogen = caching_args.allow_autogen
        self.__cache_log_probs = caching_args.cache_log_probs
        self.__cache_scores = caching_args.cache_scores

    def load_dataset(
        self,
        type,
        length,
        *,
        include_samples=True,
        include_log_probs=False,
        include_scores=False,
        **kwargs,
    ):
        """Load from cached repo assuming a specific layout"""
        T_old = self._T
        T_new = kwargs.get("T", T_old)  # Get new temperature if specified
        kB_T_new = self.__kB_T * (T_new / T_old)  # get scaled constant

        dset_cfg = _get_dataset_config_from_cached_repo(self.__repo, type=type, T=T_new)
        remote_samples_fpath = dset_cfg.get("samples")
        samples_fpath = self.__repo.load_file(remote_samples_fpath)
        samples: np.ndarray = np.load(samples_fpath)

        if samples.shape[0] < length:
            raise ValueError(
                f"Requested more samples ({length}) than available ({samples.shape[0]})"
            )
        samples = samples[:length]

        if include_log_probs:
            log_probs_fpath = self.__repo.try_load_file(dset_cfg.get("log_probs"))
            log_probs_cache_fpath = _get_cache_path(
                samples_fpath, cache_data_type="log_probs"
            )

            log_probs = cache_load_sample_derived_data(
                samples=samples,
                data_fpath=log_probs_fpath,
                data_cache_fpath=log_probs_cache_fpath,
                data_eval_fn=self.__log_prob_fn,
                allow_autogen=self.__allow_autogen,
                cache_data=self.__cache_log_probs,
            )
        else:
            log_probs = None

        if include_scores:
            scores_fpath = self.__repo.try_load_file(dset_cfg.get("scores"))
            scores_cache_fpath = _get_cache_path(
                samples_fpath, cache_data_type="scores"
            )
            scores = cache_load_sample_derived_data(
                samples,
                data_fpath=scores_fpath,
                data_cache_fpath=scores_cache_fpath,
                data_eval_fn=self.__score_fn,
                allow_autogen=self.__allow_autogen,
                cache_data=self.__cache_scores,
            )
        else:
            scores = None

        if not include_samples:
            samples = None

        return Dataset(
            kB_T=kB_T_new, samples=samples, log_probs=log_probs, scores=scores
        )


class DomainScaledDatasetLoader(DatasetLoader):
    def __init__(self, dataset_loader: DatasetLoader, length_scale: float):
        super().__init__()
        self._loader = dataset_loader
        self._length_scale = length_scale

    def load_dataset(
        self,
        type,
        length,
        *,
        include_samples=True,
        include_log_probs=False,
        include_scores=False,
        **kwargs,
    ):
        return self._loader.load_dataset(
            type=type,
            length=length,
            include_samples=include_samples,
            include_log_probs=include_log_probs,
            include_scores=include_scores,
            **kwargs,
        ).scale_domain(self._length_scale)
