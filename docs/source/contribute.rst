Contribute
===========

Adding new targets
------------------

A target should be structured as a (sub-)Python package as follows:

::

    target_package_name/
    ├── __init__.py
    │   contains:
    │       from .impl_module import TargetClassName
    │       __all__ = ["TargetClassName"]
    └── impl_module.py
        contains the class implementation TargetClassName

Templates to create targets can be found under::

https://github.com/ChristophervonKlitzing/boltzkit/tree/main/templates/targets

Targets can be implemented using either:

- :class:`boltzkit.targets.base.base.NumPyTarget`
- :class:`boltzkit.targets.base.base.DispatchedTarget`


NumPy Target
~~~~~~~~~~~~

- Log-prob and optional score implementation in NumPy
- Framework conversion (PyTorch / JAX) is handled automatically
- Best choice for targets that use a simulation framework such as OpenMM to evaluate energies and forces


Dispatched Target
~~~~~~~~~~~~~~~~~

- Log-prob and optional score implementation for every framework (NumPy, JAX, PyTorch)
- Evaluation is automatically dispatched based on the type of the input
- Best for simple targets where log-probabilities and scores can be easily implemented manually, such as Gaussian mixtures

**WARNING:** This target requires multiple implementations of the same mathematical function in different frameworks.
Make sure that they are consistent (e.g., through unit tests)!

Adding evaluation nodes
-----------------------