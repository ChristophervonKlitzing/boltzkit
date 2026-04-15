"""
============================================================================

This file utilizes a modified version of the ZMatrixFactory from
https://github.com/noegroup/bgmol.

============================================================================

MIT License

Copyright (c) 2020 noegroup

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

============================================================================
"""

import os
import warnings
from copy import copy

import numpy as np
import mdtraj as md
import yaml

from typing import Sequence

_THIS_MODULE_DIR = os.path.dirname(__file__)
_TEMPLATE_DIR = os.path.join(_THIS_MODULE_DIR, "z_matrix_templates")


def find_rings(mdtraj_topology: md.Topology):
    """Find rings in a molecule.

    Returns
    -------
    ring_atoms: List[List[int]]
        Atoms constituting each ring

    Notes
    -----
    Requires networkx.
    """
    import networkx as nx

    graph = mdtraj_topology.to_bondgraph()
    cycles = nx.cycle_basis(graph)
    return [[atom.index for atom in ring] for ring in cycles]


def is_ring_torsion(torsions: Sequence[Sequence[int]], mdtraj_topology: md.Topology):
    """Whether torsions are part of defining a ring.

    Returns
    -------
    is_ring : np.ndarray
        A boolean array which contains 1 for torsions of ring atoms and 0 for others

    Notes
    -----
    Requires networkx.
    """
    is_ring = np.zeros(len(torsions), dtype=bool)
    rings = find_rings(mdtraj_topology)
    for i, torsion in enumerate(torsions):
        # checking whether the atom that is placed by this torsion is part of any rings
        ring_indices = [j for j, ring in enumerate(rings) if torsion[0] in ring]
        if len(ring_indices) == 0:
            continue
        # checking whether any other atom in this torsion is part of the same ring
        for ring_index in ring_indices:
            if any(atom in rings[ring_index] for atom in torsion[1:]):
                is_ring[i] = True
                continue
    return is_ring


def is_proper_torsion(torsions: Sequence[Sequence[int]], mdtraj_topology: md.Topology):
    """Whether torsions are proper or improper dihedrals.

    Parameters
    ----------
    torsions : np.ndarray or Sequence[Sequence[int]]
        A list of torsions or a zmatrix.
    mdtraj_topology : md.Topology

    Returns
    -------
    is_proper : np.ndarray
        A boolean array which contains 1 for proper and 0 for improper torsions.
    """
    is_proper = np.zeros(len(torsions), dtype=bool)
    graph = mdtraj_topology.to_bondgraph()
    for i, torsion in enumerate(torsions):
        if -1 in torsion:
            continue
        atoms = [mdtraj_topology.atom(torsion[i]) for i in range(4)]
        if (
            atoms[1] in graph.neighbors(atoms[0])
            and atoms[2] in graph.neighbors(atoms[1])
            and atoms[3] in graph.neighbors(atoms[2])
        ):
            is_proper[i] = True
    return is_proper


def is_methyl_torsion(
    torsions: Sequence[Sequence[int]],
    mdtraj_topology: md.Topology,
    general: bool = False,
):
    """Whether torsions are the first (proper) torsion of a methyl group.
    Methyl hydrogens are placed by one proper and two improper torsions. This function only indicates the former.

    Parameters
    ----------
    torsions : np.ndarray or Sequence[Sequence[int]]
        A list of torsions or a zmatrix.
    mdtraj_topology : md.Topology
    general : bool
        If true any group containing three H atoms is selected, not just methyl.

    Returns
    -------
    is_methyl : np.ndarray
        A boolean array which contains 1 for proper methyl torsion and 0 otherwise
    """
    is_methyl = np.zeros(len(torsions), dtype=bool)
    is_proper = is_proper_torsion(torsions, mdtraj_topology)
    graph = mdtraj_topology.to_bondgraph()
    for i, (torsion, proper) in enumerate(zip(torsions, is_proper)):
        if not proper:
            continue
        atom = mdtraj_topology.atom(torsion[0])
        if not atom.element.symbol == "H":
            continue
        neighbor = list(graph.neighbors(atom))[0]
        if not general and not neighbor.element.symbol == "C":
            continue
        carbon_neighbors = graph.neighbors(neighbor)
        n_hydrogens = sum(n.element.symbol == "H" for n in carbon_neighbors)
        if n_hydrogens == 3:
            is_methyl[i] = True
    return is_methyl


def _select_ha(mdtraj_topology: md.Topology):
    halpha = mdtraj_topology.select("name HA")
    graph = mdtraj_topology.to_bondgraph()
    indices = []
    for ha in halpha:
        atom = mdtraj_topology.atom(ha)
        neighbors = list(graph.neighbors(atom))
        if len(neighbors) != 1:
            continue
        ca = neighbors[0]
        if ca.name != "CA":
            continue
        neighbors = list(graph.neighbors(ca))
        if all(neighbor.element.symbol != "C" for neighbor in neighbors):
            continue
        if all(neighbor.element.symbol != "N" for neighbor in neighbors):
            continue
        indices.append(ha)
    return np.array(indices)


def is_chiral_torsion(torsions: Sequence[Sequence[int]], mdtraj_topology: md.Topology):
    """True for all torsions that define HA positions."""
    halpha = _select_ha(mdtraj_topology)
    is_ha = np.zeros(len(torsions), dtype=bool)
    for i, torsion in enumerate(torsions):
        if torsion[0] in halpha:
            is_ha[i] = True
    return is_ha


def is_type_torsion(
    type_torsion: str, torsions: Sequence[Sequence[int]], mdtraj_topology: md.Topology
):
    """Whether torsions are of the specified type.
    Types supported are: ramachandran, phi, psi, omega, chi, chi1, chi2, chi3, chi4, ring, proper, methyl, chiral

    Parameters
    ----------
    type_torsion : str
        Which type of torsion among the supported ones.
    torsions : np.ndarray or Sequence[Sequence[int]]
        A list of torsions or a zmatrix.
    mdtraj_topology : md.Topology

    Returns
    -------
    is_type : np.ndarray
        A boolean array which contains True iff the torsion is of the specified type.
    """

    fake_traj = md.Trajectory(np.zeros((mdtraj_topology.n_atoms, 3)), mdtraj_topology)
    if type_torsion == "ramachandran":
        indices = np.vstack(
            (md.compute_phi(fake_traj)[0], md.compute_psi(fake_traj)[0])
        )
    elif type_torsion == "phi":
        indices = md.compute_phi(fake_traj)[0]
    elif type_torsion == "psi":
        indices = md.compute_psi(fake_traj)[0]
    elif type_torsion == "omega":
        indices = md.compute_omega(fake_traj)[0]
    elif type_torsion == "chi":
        indices = np.vstack(
            (
                md.compute_chi1(fake_traj)[0],
                md.compute_chi2(fake_traj)[0],
                md.compute_chi3(fake_traj)[0],
                md.compute_chi4(fake_traj)[0],
            )
        )
    elif type_torsion == "chi1":
        indices = md.compute_chi1(fake_traj)[0]
    elif type_torsion == "chi2":
        indices = md.compute_chi2(fake_traj)[0]
    elif type_torsion == "chi3":
        indices = md.compute_chi3(fake_traj)[0]
    elif type_torsion == "chi4":
        indices = md.compute_chi4(fake_traj)[0]
    elif type_torsion == "ring":
        return is_ring_torsion(torsions, mdtraj_topology)
    elif type_torsion == "proper":
        return is_proper_torsion(torsions, mdtraj_topology)
    elif type_torsion == "methyl":
        return is_methyl_torsion(torsions, mdtraj_topology)
    elif type_torsion == "chiral":
        return is_chiral_torsion(torsions, mdtraj_topology)
    else:
        raise ValueError(
            "Supported torsion types are: "
            "'ramachandran', 'phi', 'psi', 'omega', 'chi', 'chi1', "
            "'chi2', 'chi3', 'chi4', 'ring', 'proper', 'methyl', 'chiral'"
        )

    ordered_torsions = np.sort(
        torsions, axis=1
    )  # make sure index ordering is same as md
    is_type = np.array(
        [np.any([np.all(j == ind) for j in indices]) for ind in ordered_torsions]
    )

    return is_type


def is_ramachandran_torsion(
    torsions: Sequence[Sequence[int]], mdtraj_topology: md.Topology
):
    """Whether torsions are Ramachandran angles or not.

    Parameters
    ----------
    torsions : np.ndarray or Sequence[Sequence[int]]
        A list of torsions or a zmatrix.
    mdtraj_topology : md.Topology

    Returns
    -------
    is_ramachandran : np.ndarray
        A boolean array which contains True iff the torsion is Ramachandran
    """

    return is_type_torsion("ramachandran", torsions, mdtraj_topology)


def rewire_chiral_torsions(
    z_matrix: np.ndarray, mdtraj_topology: md.Topology, verbose=True
):
    """Redefine all torsions containing HA as HA-CA-N-C torsions.
    We use those to define chirality of aminoacids.

    Parameters
    ----------
    z_matrix : np.ndarray
        A matrix of atom indices defining the internal coordinate transform.
    mdtraj_topology : md.Topology

    Returns
    -------
    z_matrix : np.ndarray
        A modified z-matrix that has all HA positions conditioned on CA, N, C

    Notes
    -----
    The indices do not correspond directly to indices
    """
    if len(z_matrix) == 0:
        return z_matrix
    halphas = _select_ha(mdtraj_topology)
    halphas = np.intersect1d(halphas, z_matrix[:, 0])

    found_torsions = []
    for ha in halphas:
        indices = []
        for i, torsion in enumerate(z_matrix):
            if ha == torsion[0]:
                if not -1 in torsion:
                    indices.append(i)
        if len(indices) != 1:
            warnings.warn(
                f"Coordinate transform's z-matrix has no unique torsion with HA (index: {ha})"
            )
            continue
        found_torsions.append((ha, indices[0]))

    for ha, torsion_index in found_torsions:
        torsion = z_matrix[torsion_index]
        chiral_torsion = [ha]
        for name in ["CA", "N", "C"]:
            atom = mdtraj_topology.atom(ha)
            selection = mdtraj_topology.select(
                f"resid {atom.residue.index} and name {name}"
            )
            assert len(selection) == 1
            chiral_torsion.append(selection[0])
        chiral_torsion = np.array(chiral_torsion)
        if (chiral_torsion == torsion).all():
            continue
        # replace torsion by chiral_torsion
        if verbose:
            print(f"replace torsion {torsion} by {chiral_torsion}")

        # remove circular dependencies
        for other_atom in chiral_torsion[1:]:
            if other_atom not in z_matrix[:, 0]:
                continue
            other_index = np.where(z_matrix[:, 0] == other_atom)[0][0]
            if ha in z_matrix[other_index, 1:]:
                replace_with_atoms_from_torsion = ha
                replaced = False
                while not replaced:
                    try:
                        replacement = np.setdiff1d(
                            z_matrix[replace_with_atoms_from_torsion],
                            z_matrix[other_index],
                        )[0]
                        replaced = True
                    except IndexError:
                        replace_with_atoms_from_torsion = z_matrix[
                            replace_with_atoms_from_torsion
                        ][1]
                        continue
                    atom_index = np.where(z_matrix[other_index] == ha)[0][0]
                    if verbose:
                        print(
                            f"- change {z_matrix[other_index]} to {replacement} at index {atom_index}"
                        )
                    z_matrix[other_index, atom_index] = replacement

        z_matrix[torsion_index] = chiral_torsion
    return z_matrix


class ZMatrixFactory:
    """Factory for internal coordinate representations.

    Parameters
    ----------
    mdtraj_topology : mdtraj.Topology
        The system's topology.
    cartesian : str or sequence of ints
        The ids of atoms that are not transformed into internal coordinates.
        This can either be a sequence of integers or a DSL selection string for mdtraj.

    Attributes
    ----------
    z_matrix : np.ndarray
        The internal coordinate definition of shape (n_atoms - n_cartesian, 4).
        For each line, the placing of atom `line[0]` is conditioned on the atoms `line[1:3]`.
        Negative entries (-1) in a line mean that the atom is a seed for the global transform.
    fixed : np.ndarray
        One-dimensional array of cartesian atom indices.
    """

    @property
    def z_matrix(self):
        return np.array(self._z).astype(np.int64)

    @z_matrix.setter
    def z_matrix(self, value):
        if isinstance(value, list):
            self._z = value
        elif isinstance(value, np.ndarray):
            self._z = value.tolist()
        else:
            raise ValueError(f"Can't set z_matrix to {value}")

    @property
    def fixed(self):
        return np.array(self._cartesian).astype(np.int64)

    def __init__(self, mdtraj_topology, cartesian=()):
        self.top = mdtraj_topology
        self._cartesian = self._select(cartesian)
        self.graph = self.top.to_bondgraph()
        self._atoms = sorted(list(self.graph.nodes), key=lambda node: node.index)
        self._distances = self._build_distance_matrix(self.graph)
        self._z = []

    # === naive builder and helpers ===

    def build_naive(
        self, subset="all", render_independent=True, rewire_chiral=True, verbose=False
    ):
        """Place atoms relative to the closest atoms that are already placed (wrt. bond topology).

        Parameters
        ----------
        subset : str or sequence of int
            A selection string or list of atoms. The z-matrix is only build for the subset.
        render_independent : bool
            Whether to make sure that no two positions depend on the same three other positions.
        rewire_chiral : bool
            Whether to make sure that all HA depend on (CA, N, C).

        Returns
        -------
        z_matrix : np.ndarray
        fixed_atoms : np.ndarray
        """
        subset = self._select(subset)
        current = set(self._placed_atoms())
        if len(current) < 3:
            self._z = self._seed_z(current, subset)
            for torsion in self._z:
                current.add(torsion[0])
        while len(current) > 0:
            new_current = copy(current)
            for atom in current:
                assert self._is_placed(atom)
                for neighbor in self._neighbors(atom):
                    if not self._is_placed(neighbor) and neighbor in subset:
                        new_current.add(neighbor)
                new_current.remove(atom)
            current = new_current
            # build part of z matrix
            z = []
            for atom in current:
                closest = self._3closest_placed_atoms(atom, subset=subset)
                if len(closest) == 3:
                    z.append([atom, *closest])
            self._z.extend(z)
        if rewire_chiral:
            self.z_matrix = rewire_chiral_torsions(
                self.z_matrix, self.top, verbose=verbose
            )
        if render_independent:
            self.render_independent(keep=_select_ha(self.top))
        return self.z_matrix, self.fixed

    @staticmethod
    def _build_distance_matrix(graph):
        import networkx as nx

        distances = nx.all_pairs_shortest_path_length(graph)
        matrix = np.zeros((len(graph), len(graph)))
        for this, dist in distances:
            for other in dist:
                matrix[this.index, other.index] = dist[other]
        return matrix

    def _seed_z(self, current, subset):
        current = list(current)
        if len(current) == 0:
            current = [subset[0]]
        closest = self._3closest_placed_atoms(current[0], subset, subset)
        seed = np.unique([*current, *closest])[:3]
        z = [
            [seed[0], -1, -1, -1],
            [seed[1], seed[0], -1, -1],
            [seed[2], seed[1], seed[0], -1],
        ]
        return z

    def _neighbors(self, i):
        for neighbor in self.graph.neighbors(self._atoms[i]):
            yield neighbor.index

    def _is_placed(self, i):
        return any(torsion[0] == i for torsion in self._z) or any(
            f == i for f in self._cartesian
        )

    def _placed_atoms(self):
        for f in self._cartesian:
            yield f
        for torsion in self._z:
            yield torsion[0]

    def _torsion_index(self, i):
        return list(self._placed_atoms()).index(i)

    def _3closest_placed_atoms(self, i, placed=None, subset=None):
        if placed is None:
            placed = np.array(list(self._placed_atoms()))
        if subset is not None:
            placed = np.intersect1d(placed, subset)
        argsort = np.argsort(self._distances[i, placed])
        closest_atoms = placed[argsort[:3]]
        return closest_atoms

    def _select(self, selection):
        return self.top.select(selection) if isinstance(selection, str) else selection

    # === template builder and helpers ===

    def build_with_templates(
        self,
        *yaml_files,
        template_lookup_dir: str = _TEMPLATE_DIR,
        build_protein_backbone=True,
        subset="all",
    ):
        """Build ICs from template files.

        Parameters
        ----------
        *yaml_files : str
            filenames of any other template files; if non are passed, use the bundled "z_protein.yaml", "z_termini.yaml"
        template_lookup_dir : str
            Directory path to look up the templates
        build_protein_backbone : bool
            Whether to build the protein backbone first
        subset : str or sequence of int
            A selection string or list of atoms. The z-matrix is only build for the subset.

        Notes
        -----
        For the formatting of template files, see data/z_protein.yaml

        Returns
        -------
        z_matrix : np.ndarray
        fixed_atoms : np.ndarray
        """

        if len(yaml_files) == 0:
            yaml_files = ["z_protein.yaml", "z_termini.yaml"]
        subset = self._select(subset)
        templates = self._load_templates(
            *yaml_files, template_lookup_dir=template_lookup_dir
        )

        # build_backbone
        if build_protein_backbone:
            self.build_naive(
                subset=np.intersect1d(
                    self.top.select("backbone and element != O"), subset
                )
            )
        # build residues
        residues = list(self.top.residues)
        for i, residue in enumerate(residues):
            is_nterm = (i == 0) and residue.is_protein
            is_cterm = ((i + 1) == self.top.n_residues) and residue.is_protein
            resatoms = {a.name: a.index for a in residue.atoms}
            if not is_cterm:
                resatoms_neighbor = {
                    f"+{a.name}": a.index for a in residues[i + 1].atoms
                }
                resatoms.update(resatoms_neighbor)
            if not is_nterm:
                resatoms_neighbor = {
                    f"-{a.name}": a.index for a in residues[i - 1].atoms
                }
                resatoms.update(resatoms_neighbor)

            # add template definitions
            definitions = templates[residue.name]
            if is_nterm and "NTERM" in templates:
                definitions = definitions + templates["NTERM"]
            if is_cterm and "CTERM" in templates:
                definitions = definitions + templates["CTERM"]

            for entry in definitions:  # template entry:
                if any(e not in resatoms for e in entry):
                    continue  # skip torsions with non-matching atom names
                if resatoms[entry[0]] in subset and not self._is_placed(
                    resatoms[entry[0]]
                ):  # not in not_ic:
                    self._z.append([resatoms[_e] for _e in entry])

        # append missing
        placed = np.array(list(self._placed_atoms()))
        if not len(placed) == len(subset):
            missing = np.setdiff1d(np.arange(self.top.n_atoms), placed)
            warnings.warn(
                f"Not all atoms found in templates. Applying naive reconstruction for missing atoms: "
                f"{tuple(self._atoms[m] for m in missing)}"
            )
            self.build_naive(subset)
        return self.z_matrix, self.fixed

    def _load_template(self, yaml_file, template_lookup_dir: str):
        filename_in_pkg = os.path.join(template_lookup_dir, yaml_file)
        if os.path.isfile(filename_in_pkg):
            if os.path.isfile(yaml_file) and os.path.normpath(
                yaml_file
            ) != os.path.normpath(filename_in_pkg):
                raise warnings.warn(
                    f"{yaml_file} exists locally and in the package templates. "
                    f"Taking the built-in one from the boltzkit package.",
                    UserWarning,
                )
            yaml_file = filename_in_pkg
        with open(yaml_file, "r") as f:
            templates = yaml.load(f, yaml.SafeLoader)
            for residue in templates:
                if not isinstance(templates[residue], list):
                    raise IOError("File format is not acceptable.")
                if "GENERAL" in templates:
                    templates[residue] = [*templates["GENERAL"], *templates[residue]]
                for torsion in templates[residue]:
                    if not len(torsion) == 4:
                        raise IOError(
                            f"Torsion {torsion} in residue {residue} does not have 4 atoms."
                        )
                    if not all(isinstance(name, str) for name in torsion):
                        raise IOError(
                            f"Torsion {torsion} in residue {residue} does not consist of atom names."
                        )
            if "GENERAL" in templates:
                del templates["GENERAL"]
            return templates

    def _load_templates(self, *yaml_files, template_lookup_dir: str):
        templates = dict()
        for f in yaml_files:
            template = self._load_template(f, template_lookup_dir=template_lookup_dir)
            for key in template:
                if key in templates:
                    warnings.warn(
                        f"Residue {key} found in multiple files. Updating with the definition from {f}."
                    )
            templates.update(template)
        return templates

    def build_with_system(self, system):
        """
        Idea: build a lookup table of torsions and impropers; sort them by marginal entropy.
        Before doing the naive lookup, try to insert in the minimum-entropy torsions.
        Maybe: also do something regarding symmetries.
        """
        raise NotImplementedError()

    def render_independent(self, keep=None):
        """Rewire z matrix so that no two positions depend on the same bonded atom and angle/torsion

        Parameters
        ----------
        keep : Sequence[int]
            All atoms, whose placement should not be changed by any means.
            By default, don't rewire CB to be able to control the chirality.
        """
        keep = _select_ha() if keep is None else keep
        keep = self._select(keep)
        all234 = [(torsion[1], set(torsion[2:])) for torsion in self._z]
        for i in range(len(self._z)):
            torsion = self._z[i]
            this234 = all234[i]
            while this234 in all234[:i]:
                previous_index = all234.index(this234)
                previous_torsion = self._z[previous_index]
                if torsion[0] in keep:
                    assert not previous_torsion[0] in keep
                    # swap
                    self._z[i], self._z[previous_index] = previous_torsion, torsion
                    all234[i], all234[previous_index] = (
                        all234[previous_index],
                        all234[i],
                    )
                    torsion = self._z[i]
                    this234 = all234[i]
                    continue
                # make sure there are no circular dependencies
                assert previous_torsion[0] not in torsion[:3]
                assert torsion[0] not in previous_torsion
                # rewire
                new_torsion = torsion[:3] + previous_torsion[:1]
                self._z[i] = new_torsion
                this234 = set(new_torsion[2:])
                all234[i] = this234
            assert not this234 in all234[:i]
            assert not self._z[i] in self._z[:i]
        if not self.is_independent(self._z):
            warnings.warn(
                "Z-matrix torsions are not fully independent because of a constraint on HA."
            )
        return self._z

    @staticmethod
    def is_independent(z):
        all234 = [(torsion[1], set(torsion[2:])) for torsion in z]
        for i, other in enumerate(all234):
            if other in all234[:i]:
                return False
        return True


def build_fake_topology(n_atoms, bonds=None, atoms_by_residue=None, coordinates=None):
    """A stupid function to build an MDtraj topology with limited information.

    Returns
    -------
    topology : md.Topology
    trajectory : md.Trajectory or None
        None, if no coordinates were specified.
    """
    topology = md.Topology()

    if bonds is None:  # assume linear molecule
        bonds = np.column_stack([np.arange(n_atoms - 1), np.arange(1, n_atoms)])
    if atoms_by_residue is None:  # 1 atom per residue
        atoms = set([atom for bond in bonds for atom in bond])
        atoms_by_residue = [[atom] for atom in atoms]

    chain = topology.add_chain()
    for res in atoms_by_residue:
        residue = topology.add_residue("C", chain)
        for _ in res:
            topology.add_atom("CA", chain, residue)
    atoms = list(topology.atoms)
    for bond in bonds:
        atom1 = atoms[bond[0]]
        atom2 = atoms[bond[1]]
        topology.add_bond(atom1, atom2)

    trajectory = None
    if coordinates is not None:
        coords = coordinates.reshape(-1, n_atoms, 3)
        trajectory = md.Trajectory(xyz=coords, topology=topology)
    return topology, trajectory
