import numpy as np
from boltzkit.targets.base.base import NumpyEval


def lennard_jones_energy_numpy(r: np.ndarray, eps=1.0, rm=1.0):
    return eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)


def tile_numpy(a: np.ndarray, dim: int, n_tile: int):
    init_dim = a.shape[dim]
    repeat_idx = [1] * a.ndim
    repeat_idx[dim] = n_tile
    a = np.tile(a, repeat_idx)

    order_index = np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
    )
    return np.take(a, order_index, axis=dim)


def distance_vectors_numpy(x: np.ndarray, remove_diagonal: bool = True):
    # x: (B, N, D)
    r = tile_numpy(np.expand_dims(x, axis=2), 2, x.shape[1])
    r = r - np.transpose(r, (0, 2, 1, 3))

    if remove_diagonal:
        B, N, _, D = r.shape
        mask = ~np.eye(N, dtype=bool)
        r = r[:, mask].reshape(B, N, N - 1, D)

    return r


def distances_from_vectors_numpy(r: np.ndarray, eps=1e-6):
    return np.sqrt(np.sum(r**2, axis=-1) + eps)


class NumpyLennardJonesEval(NumpyEval):
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
        self.n_particles = n_particles
        self.spatial_dims = spatial_dims
        self.dim = n_particles * spatial_dims

        self.eps = eps
        self.rm = rm
        self.oscillator = oscillator
        self.oscillator_scale = oscillator_scale
        self.energy_factor = energy_factor

    def _remove_mean(self, x: np.ndarray):
        # x: (B, N, D)
        return x - np.mean(x, axis=1, keepdims=True)

    def _energy(self, x: np.ndarray):
        """
        x: (B, dim)
        returns: (B,)
        """
        if x.ndim == 1:
            x = x[None, :]

        B = x.shape[0]

        # reshape to (B, N, D)
        x = x.reshape(B, self.n_particles, self.spatial_dims)

        # pairwise distances
        r_vec = distance_vectors_numpy(x)  # (B, N, N-1, D)
        dists = distances_from_vectors_numpy(r_vec)  # (B, N, N-1)

        # LJ energy
        lj = lennard_jones_energy_numpy(dists, self.eps, self.rm)
        lj_energy = np.sum(lj.reshape(B, -1), axis=1) * self.energy_factor

        # oscillator term
        if self.oscillator:
            centered = self._remove_mean(x)
            osc = 0.5 * np.sum(centered**2, axis=(1, 2))
            lj_energy = lj_energy + self.oscillator_scale * osc

        return lj_energy  # (B,)

    def get_log_prob(self, x):
        """
        x: (B, D)
        returns: (B,)
        """
        return -self._energy(x)

    def get_score(self, x):
        raise NotImplementedError

    def get_log_prob_and_score(self, x):
        raise NotImplementedError
