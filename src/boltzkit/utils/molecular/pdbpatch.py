"""
Copied from https://github.com/noegroup/bgmol/blob/main/bgmol/util/pdbpatch.py

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
"""

from contextlib import contextmanager
from openmm import app


@contextmanager
def fixed_atom_names(**residues):
    """Suppress replacing of selected atom names when creating PSF or PDB files.

    Examples
    --------
    >>> with fixed_atom_names(ALA=["NT"]):
    >>>     psf = app.CharmmPsfFile("some.psf")
    """

    def modify(original):
        @staticmethod
        def modified_load_tables():
            original()
            for residue in residues:
                for atom in residues[residue]:
                    if atom in app.PDBFile._atomNameReplacements[residue]:
                        del app.PDBFile._atomNameReplacements[residue][atom]

        return modified_load_tables

    app.PDBFile._loadNameReplacementTables, original = (
        modify(app.PDBFile._loadNameReplacementTables),
        app.PDBFile._loadNameReplacementTables,
    )
    yield
    app.PDBFile._loadNameReplacementTables = original
