"""
This implementation is copied and from
'https://github.com/aimat-lab/AnnealedBG/tree/adaptive_smoothing'.

----------------------------------------------------------------------------

MIT License

Copyright (c) 2025 Henrik Schopmans

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

----------------------------------------------------------------------------
"""

import mdtraj as md
import numpy as np

SELECTION = "symbol == C or symbol == N or symbol == S"


def _wrap(array: np.ndarray):
    return (np.sin(array), np.cos(array))


def _distances(xyz: np.ndarray):
    distance_matrix_ca: np.ndarray = np.linalg.norm(
        xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1
    )
    n_ca = distance_matrix_ca.shape[-1]
    m, n = np.triu_indices(n_ca, k=1)
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca


def _tica_features(
    trajectory: md.Trajectory,
    use_dihedrals: bool = True,
    use_distances: bool = True,
    selection: str = SELECTION,
):
    top: md.Topology = trajectory.topology
    trajectory = trajectory.atom_slice(top.select(selection))
    # n_atoms = trajectory.xyz.shape[1]
    if use_dihedrals:
        _, phi = md.compute_phi(trajectory)
        _, psi = md.compute_phi(trajectory)
        _, omega = md.compute_omega(trajectory)
        dihedrals = np.concatenate([*_wrap(phi), *_wrap(psi), *_wrap(omega)], axis=-1)
    if use_distances:
        ca_distances = _distances(trajectory.xyz)
    if use_distances and use_dihedrals:
        return np.concatenate([ca_distances, dihedrals], axis=-1)
    elif use_distances:
        return ca_distances
    else:
        return []
