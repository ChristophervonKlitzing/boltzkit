# boltzkit
Python package for molecular Boltzmann and related densities. The package provides the target systems, utility for loading datasets, evaluation pipelines, chirality filtering, ...


# Setup
An environment with all dependencies can be installed in the following way:
```bash
conda env create -f environment.yaml
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