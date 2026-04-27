Molecular Boltzmann Targets
===========================

TODO: Update API to create MolecularBoltzmann targets to use classmethods instead
TODO: The API on how to create MolecularBoltzmann targets will likely change soon

Overview
--------

The :class:`boltzkit.targets.boltzmann.MolecularBoltzmann` class defines a family of molecular energy-based targets representing Boltzmann distributions over molecular conformations.

Targets can be initialized from multiple sources:

- HuggingFace datasets (predefined systems such as alanine dipeptides/peptides)
- Local directories
- PDB files

Instantiation
-------------

A ``MolecularBoltzmann`` target can be created as follows:

.. code-block:: python

   from boltzkit.targets.boltzmann import MolecularBoltzmann

   target = MolecularBoltzmann("datasets/chrklitz99/alanine_dipeptide")


Supported input sources
-----------------------

HuggingFace datasets
~~~~~~~~~~~~~~~~~~~~~

Predefined molecular systems are available via HuggingFace:

- ``datasets/chrklitz99/alanine_dipeptide``
- ``datasets/chrklitz99/alanine_tetrapeptide``
- ``datasets/chrklitz99/alanine_hexapeptide``

These datasets provide both specification of the forcefields and ready-to-use high-quality MD data.

Local directory
~~~~~~~~~~~~~~~~

A target can also be initialized from a local dataset directory:

.. code-block:: python

   target = MolecularBoltzmann("/path/to/local/dataset")

The directory must contain the required files in the expected format similar to the huggingface repositories.

PDB file
~~~~~~~~

For quick setup from a single molecular structure, a PDB file can be used:

.. code-block:: python

   target = MolecularBoltzmann.create_from_pdb("custom_target_name", "structure.pdb")


Common operations
-----------------

Once initialized, the target provides dataset utilities and molecular metadata:

.. code-block:: python

   val_dataset = target.load_dataset(T=300.0, type="val") # if available
   topology = target.get_mdtraj_topology()
   tica_model = target.get_tica_model() # if available


Evaluation functions:

.. code-block:: python

   target.get_log_prob(samples)
   target.get_score(samples)
   target.get_log_prob_and_score(samples)


Length scale configuration
--------------------------

Internally, all MolecularBoltzmann targets operate in nanometers. Coordinate units can be configured at initialization.

.. code-block:: python

   target = MolecularBoltzmann(
       "datasets/chrklitz99/alanine_dipeptide",
       length_unit="angstrom"  # default: "nanometer"
   )


Unit handling
~~~~~~~~~~~~~~

When a length unit is specified, all inputs and outputs (coordinates, forces, scores) are automatically converted to and from that unit.

This ensures consistent usage across the entire API without manual conversions.

We strongly recommend using a consistent unit system across all evaluations involving MolecularBoltzmann targets.


Alternative specification
~~~~~~~~~~~~~~~~~~~~~~~~~

A scalar scaling factor can also be provided:

- ``"nanometer"`` → ``1.0``
- ``"angstrom"`` → ``0.1``