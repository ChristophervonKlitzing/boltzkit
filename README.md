![Tests (main)](https://img.shields.io/github/actions/workflow/status/ChristophervonKlitzing/boltzkit/run-tests.yaml?branch=main&label=tests%20(main))
![Tests (dev)](https://img.shields.io/github/actions/workflow/status/ChristophervonKlitzing/boltzkit/run-tests.yaml?branch=dev&label=tests%20(dev))
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://christophervonklitzing.github.io/boltzkit/)

# The boltzkit package
`boltzkit` is a Python package for working with molecular Boltzmann and related densities.
It provides tools for defining target systems, loading datasets, running evaluation pipelines, and performing tasks such as chirality filtering.

The package is designed to make it as simple as possible to:
- evaluate target energies / log probabilities
- compute forces / scores
- evaluate predicted samples

## Documentation
Full documentation (installation, tutorials, and API reference):

👉 https://christophervonklitzing.github.io/boltzkit/

**WARNING (TODO):** boltzkit is currently not in the state where targets should be added. A smaller refactoring is needed before that
to split density evaluation, sampling, properties like the presennce of a log normalization constant, loading of datasets etc into separate interface classes. A target should no longer be an explicit base-class but rather a composition of interfaces like `AdmitsDensity`, `HasDataset`, `HasKnownNormalizer`, `AdmitsSampling`. To check functionality, use `isinstance(target, AdmitsSampling)` instead of methods like `target.can_sample()`. To document these properties, a table should be created in the documentation. Certain interfaces can also have deriving implementations such as `HasCachedRepoDataset` to easily allow for pre-implemented routines.


# TODOs
- Chirality filtering
- Test data augmentation
