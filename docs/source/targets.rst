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


Adding New Targets to `boltzkit`
---------------------------------

Targets can be implemented using either:

- :class:`boltzkit.targets.base.base.NumPyTarget`
- :class:`boltzkit.targets.base.base.DispatchedTarget`

1. **NumPy Target** 

- Log-prob and optional score implementation in NumPy
- Framework conversion (PyTorch / JAX) is handled automatically
- Best choice for targets that use a simulation framework such as OpenMM to evaluate energies and forces

.. literalinclude:: ../../demos/create_numpy_target.py
    :language: python
    :linenos:


2. **Dispatched Target**

- Log-prob and optional score implementation for every framework (NumPy, Jax, PyTorch)
- Evaluation is automatically dispatched based on the type of the input
- Best for simple targets where log-probabilities and scores can be easily implemented manually, such as Gaussian mixtures

**WARNING: This target requires multiple implementations of the same mathematical function in different frameworks. 
Make sure that they are consistent (e.g., through unit-tests)!**

.. literalinclude:: ../../demos/create_dispatched_target.py
    :language: python
    :linenos:


