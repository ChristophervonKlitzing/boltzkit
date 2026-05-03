Target systems
==============

Overview
--------

All target systems implement the following interface:

- ``get_log_prob``
- ``get_score``
- ``get_log_prob_and_score``

These methods accept (batched) inputs as NumPy arrays, JAX arrays, or PyTorch tensors and return outputs in the corresponding framework format, respecting dtypes and device where necessary.

Each target defines which operations are supported and which backend frameworks are available.

Available targets
------------------

- :doc:`targets/molecular_boltzmann`
- :doc:`targets/gaussian_mixture`
- :doc:`targets/lennard_jones`

