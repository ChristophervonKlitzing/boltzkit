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


A target is a composition of capability providers that define how it behaves,
for example whether it can evaluate densities or provide datasets. The minimal requirement
is to inherit from :class:`~boltzkit.targets.base.base.BaseTarget`.

Density providers
~~~~~~~~~~~~~~~~~~~~~~~~~

Define how a target evaluates probabilities and scores.

- :class:`~boltzkit.targets.base.density_provider.NumpyDensityProvider`
  NumPy-based implementation of `log_prob` and score functions.

- :class:`~boltzkit.targets.base.density_provider.DispatchedDensityProvider`
  Multi-backend implementation supporting NumPy, PyTorch, and JAX.

Dataset providers
~~~~~~~~~~~~~~~~~~~~~~~~~

Define how a target provides or constructs datasets.

- :class:`~boltzkit.targets.base.dataset_provider.ExternalDatasetProvider`
  Uses a manually specified dataloader such as
  :class:`~boltzkit.utils.dataloader.CachedRepoDatasetLoader`
  to load datasets from external sources like Hugging Face.

- :class:`~boltzkit.targets.base.dataset_provider.ProceduralDatasetProvider`
  Generates datasets procedurally (e.g. Gaussian mixture models).



Adding evaluation nodes
-----------------------