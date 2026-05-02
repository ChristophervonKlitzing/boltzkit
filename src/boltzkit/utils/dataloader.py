from functools import lru_cache
from pathlib import Path
import pickle
from typing import Callable, Literal

import numpy as np
import openmm.app as app
import mdtraj as md


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
    :param data_eval_fn: Function to compute derived data from samples.
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

        if data.shape[0] != n_samples:
            raise ValueError(
                f"Loaded data length mismatch: expected {n_samples} samples, "
                f"but got {data.shape[0]} from '{data_fpath}'."
            )

    if cache_data and not allow_autogen:
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
