from functools import lru_cache
from pathlib import Path
import pickle
from typing import Literal

import numpy as np
import openmm.app as app
import mdtraj as md


@lru_cache
def load_from_file(
    path: str | Path,
    data_type: Literal["log_probs", "samples"],
    n_samples: int | None = None,
    dtype: type | np.dtype = np.float64,
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

    data = np.array(data, dtype=dtype)
    return data


def load_tica_model(path: str | Path):
    # TODO: Also allow numpy and pytorch arrays and convert to deeptime tica model
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"TICA model file does not exist: {path}")

    with open(path, "rb") as f:
        tica_model = pickle.load(f)

    return tica_model


def load_topology(path: str | Path):
    # TODO: Support other formats

    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Topology file does not exist: {path}")

    if not path.suffix.lower().endswith(".pdb"):
        raise RuntimeError(f"Could not load topology from path '{path}'")

    mm_top = app.PDBFile(path.as_posix()).topology
    return md.Topology.from_openmm(mm_top)
