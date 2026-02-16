# boltzkit
Python package for molecular Boltzmann and related densities. The package provides the target systems, utility for loading datasets, evaluation pipelines, chirality filtering, ...


# Setup
An environment with all dependencies can be installed in the following way:
```bash
conda env create -f environment.yaml
```

To activate the environment:
```bash
conda activate boltzkit
```

# Testing
## Run unit tests
To run all unittests, run:
```bash
python -m unittest
```
Alternatively, specific unittests can be executed with:
```bash
python -m unittest tests.demo
```

## Create unit test
test files need to start with `test_` to be discoverable.


# Project structure
- `boltzkit.evaluation`: density & energy-based evaluation code
- `boltzkit.molecular`: Molecular system loading, creation, and energy-, and density-evaluation
- `boltzkit.targets`: Framework-agnostic (PyTorch, Jax, NumPy) wrapper classes for common molecular Boltzmann densities and other toy systems (e.g., Lennard Jones, GMM40, ...)
- `boltzkit.utils`: Data augmentation, chirality filtering, as well as general utility functions (e.g., dataset-loader, decorators to make functions framework agnostic, ...)
--- 
- `tools`: General ready-to-use tools based on the boltzkit package (e.g., MD dataset creation (for molecular systems), dataset-downloader, evaluation-tool based on file inputs, ...). It is not part of the boltzkit package because it contains ready-to-use applications. 