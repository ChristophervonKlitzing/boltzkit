import math
from typing import Optional, Callable, TYPE_CHECKING
import numpy as np
from scipy.spatial.transform import Rotation as R
from boltzkit.utils.framework import create_dispatch

if TYPE_CHECKING:
    from boltzkit.utils.framework import Array
    import torch


# TODO: Add seed argument and use numpy rng for both implementations
# TODO: Add test with if __name__ == "__main__"


def create_symmetry_augmentation(
    sigma: Optional[float] = None,
    rotation_augmentation: bool = True,
    COM_augmentation: bool = True,
) -> Callable[["Array", bool | None, bool | None, bool | None], "Array"]:
    """
    Create a rotation and center-of-mass translation augmentation function.

    This factory returns a callable that applies stochastic rigid-body
    transformations to molecular coordinate samples. The augmentation consists of:

    1. Removing the center-of-mass (COM) from each sample.
    2. Optionally applying a random 3D rotation.
    3. Optionally applying a random COM translation sampled from a Gaussian.

    The function supports both NumPy arrays and PyTorch tensors via a
    framework dispatch mechanism.

    Parameters
    ----------
    sigma : float or None, optional
        Standard deviation of the Gaussian COM translation. If ``None``,
        the default value ``1 / sqrt(n_atoms)`` is used for each sample,
        where ``n_atoms`` is inferred from the input dimensionality.
    rotation_augmentation : bool, default=True
        Whether to apply random 3D rotations to each sample.
    COM_augmentation : bool, default=True
        Whether to apply a random center-of-mass translation after removing
        the original COM. If ``False``, samples remain centered at the origin.

    Returns
    -------
    Callable
        A function ``augment(samples, sigma_override=None,
        rotation_override=None, COM_override=None)`` that applies the
        augmentation. The function accepts both NumPy arrays and PyTorch
        tensors and returns an array of the same type.

    Notes
    -----
    The input samples are assumed to represent flattened 3D coordinates
    of atoms in the form::

        (batch_size, n_atoms * 3)

    Internally the coordinates are reshaped to::

        (batch_size, n_atoms, 3)

    before applying the transformations.

    This augmentation strategy is inspired by rigid-body data augmentation
    used in molecular generative modeling, e.g. in
    *Scalable Equilibrium Sampling with Sequential Boltzmann Generators*.
    """

    def augment_torch(
        samples: "torch.Tensor",
        sigma_override: Optional[float] = None,
        rotation_override: Optional[bool] = None,
        COM_override: Optional[bool] = None,
    ) -> "torch.Tensor":

        assert len(samples.shape) == 2
        batch, samples_dim = samples.shape

        spatial_dim = 3
        assert samples_dim % spatial_dim == 0

        n_atoms = samples_dim // spatial_dim

        sigma_local = sigma_override if sigma_override is not None else sigma
        if sigma_local is None:
            sigma_local = 1 / math.sqrt(n_atoms)

        rotation_enabled = (
            rotation_override
            if rotation_override is not None
            else rotation_augmentation
        )
        COM_enabled = COM_override if COM_override is not None else COM_augmentation

        samples_ = samples.reshape((batch, n_atoms, spatial_dim))

        center_of_mass = samples_.mean(dim=1, keepdim=True)
        samples_ = samples_ - center_of_mass

        if rotation_enabled:
            rotation = torch.tensor(R.random(batch).as_matrix()).to(samples_)
            samples_ = torch.einsum("bij,bki->bkj", rotation, samples_)

        if COM_enabled:
            translation = torch.randn((batch, 1, 3)).to(samples_) * sigma_local
            samples_ = samples_ + translation

        return samples_.flatten(1)

    def augment_np(
        samples: np.ndarray,
        sigma_override: Optional[float] = None,
        rotation_override: Optional[bool] = None,
        COM_override: Optional[bool] = None,
    ) -> np.ndarray:

        assert len(samples.shape) == 2
        batch, samples_dim = samples.shape

        spatial_dim = 3
        assert samples_dim % spatial_dim == 0

        n_atoms = samples_dim // spatial_dim

        sigma_local = sigma_override if sigma_override is not None else sigma
        if sigma_local is None:
            sigma_local = 1 / math.sqrt(n_atoms)

        rotation_enabled = (
            rotation_override
            if rotation_override is not None
            else rotation_augmentation
        )
        COM_enabled = COM_override if COM_override is not None else COM_augmentation

        samples_ = samples.reshape((batch, n_atoms, spatial_dim))

        center_of_mass = samples_.mean(dim=1, keepdims=True)
        samples_ = samples_ - center_of_mass

        if rotation_enabled:
            rotation = np.ndarray(R.random(batch).as_matrix())
            samples_: np.ndarray = np.einsum("bij,bki->bkj", rotation, samples_)

        if COM_enabled:
            translation = np.random.randn((batch, 1, 3)) * sigma_local
            samples_ = samples_ + translation

        return samples_.reshape((batch, -1))

    return create_dispatch(impl_torch=augment_torch, impl_np=augment_np)
