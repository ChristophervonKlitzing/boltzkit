import numpy as np
import torch
from boltzkit.targets.base.dispatched_eval.torch import TorchEval


def lennard_jones_energy_torch(r: torch.Tensor, eps: float = 1.0, rm: float = 1.0):
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


class LennardJonesPotential:
    def __init__(
        self,
        n_particles: int,
        spatial_dims: int = 3,
        eps=1.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0,
        energy_factor=1.0,
    ):
        """Energy for a Lennard-Jones cluster.

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        """

        self._n_particles = n_particles
        self._spatial_dims = spatial_dims

        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale

        # this is to match the eacf energy with the eq-fm energy
        # for lj13, to match the eacf set energy_factor=0.5
        self._energy_factor = energy_factor

    def energy(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.view(1, -1)
        # else:
        batch_shape = x.shape[0]
        x = x.view(batch_shape, self._n_particles, self._spatial_dims)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._spatial_dims))
        )

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        # lj_energies = torch.clip(lj_energies, -1e4, 1e4)
        lj_energies = (
            lj_energies.view(batch_shape, -1).sum(dim=-1) * self._energy_factor
        )

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)).view(
                batch_shape
            )
            lj_energies = lj_energies + osc_energies * self._oscillator_scale

        return lj_energies[:, None]

    def _remove_mean(self, x: torch.Tensor):
        x = x.view(-1, self._n_particles, self._spatial_dims)
        return x - torch.mean(x, dim=1, keepdim=True)

    def log_prob(self, x):
        return -self.energy(x)


class TorchLennardJonesEval(TorchEval):
    def __init__(
        self,
        n_particles: int,
        spatial_dims: int = 3,
        energy_factor=1.0,
    ):
        super().__init__()

        self.n_particles = n_particles
        self.spatial_dims = spatial_dims

        self.lennard_jones = LennardJonesPotential(
            n_particles=n_particles,
            spatial_dims=spatial_dims,
            eps=1.0,
            rm=1.0,
            oscillator=True,
            oscillator_scale=1.0,
            energy_factor=energy_factor,
        )

    @property
    def dim(self) -> int:
        return self.n_particles * self.spatial_dims

    def _get_log_prob(self, x):
        return self.lennard_jones.log_prob(x).squeeze(-1).squeeze(-1)


def distance_vectors(x: torch.Tensor, remove_diagonal: bool = True):
    r"""
    Computes the matrix :math:`r` of all distance vectors between
    given input points where

    .. math::
        r_{ij} = x_{i} - y_{j}

    as used in :footcite:`Khler2020EquivariantFE`

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    remove_diagonal : boolean
        Flag indicating whether the all-zero distance vectors
        `x_i - x_i` should be included in the result

    Returns
    -------
    r : torch.Tensor
        Matrix of all distance vectors r.
        If `remove_diagonal=True` this is a tensor of shape
            `[n_batch, n_particles, n_particles, n_dimensions]`.
        Otherwise this is a tensor of shape
            `[n_batch, n_particles, n_particles - 1, n_dimensions]`.

    References
    ----------
    .. footbibliography::

    """
    r = tile(x.unsqueeze(2), 2, x.shape[1])
    r = r - r.permute([0, 2, 1, 3])
    if remove_diagonal:
        r = r[:, torch.eye(x.shape[1], x.shape[1]) == 0].view(
            -1, x.shape[1], x.shape[1] - 1, x.shape[2]
        )
    return r


def distances_from_vectors(r: torch.Tensor, eps: float = 1e-6):
    """
    Computes the all-distance matrix from given distance vectors.

    Parameters
    ----------
    r : torch.Tensor
        Matrix of all distance vectors r.
        Tensor of shape `[n_batch, n_particles, n_other_particles, n_dimensions]`
    eps : Small real number.
        Regularizer to avoid division by zero.

    Returns
    -------
    d : torch.Tensor
        All-distance matrix d.
        Tensor of shape `[n_batch, n_particles, n_other_particles]`.
    """
    return (r.pow(2).sum(dim=-1) + eps).sqrt()


def tile(a: torch.Tensor, dim: int, n_tile: int):
    """
    Tiles a pytorch tensor along one an arbitrary dimension.

    Parameters
    ----------
    a : PyTorch tensor
        the tensor which is to be tiled
    dim : Integer
        dimension along the tensor is tiled
    n_tile : Integer
        number of tiles

    Returns
    -------
    b : PyTorch tensor
        the tensor with dimension `dim` tiled `n_tile` times
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
    )
    order_index = torch.LongTensor(order_index).to(a).long()
    return torch.index_select(a, dim, order_index)


if __name__ == "__main__":
    torch.random.manual_seed(0)

    lj = TorchLennardJonesEval(n_particles=13)

    x = torch.randn((2, lj.dim))
    lp1 = lj.get_log_prob(x)
    score1 = lj.get_score(x)

    print(lp1)
    print(score1)
