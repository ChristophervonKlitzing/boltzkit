# boltzkit
Python package for molecular Boltzmann and related densities. The package provides the target systems, utility for loading datasets, evaluation pipelines, chirality filtering, ...

## Framework-agnostic
`boltzkit` relies on NumPy wherever possible and deliberately avoids depending directly on frameworks like JAX or PyTorch, while still offering optional wrappers for them. This design ensures maximum flexibility and broad compatibility with any NumPy-based or NumPy-compatible machine learning framework.


# Setup

## Installation
Run 
```bash
pip install git+https://github.com/ChristophervonKlitzing/boltzkit
```


## Development of boltzkit
An environment with all dependencies can be installed in the following way:
```bash
conda env create -f environment.yaml
```

To activate the environment:
```bash
conda activate boltzkit
```





# Project structure
`boltzkit` is broadly separated into the sub-packages:
- `evaluation`
- `targets`
- `utils`

While `evaluation` and `targets` can be used completely indpendently, both depend on `utils`. 

--- 

- `boltzkit.evaluation`: sampling & density-based evaluation code, as well es domain-specific evaluation code. "density-based" means that a model density is available (importance weights are possible). "sampling-based" means that no importance weights are required (target density evaluations still are!).
- `boltzkit.targets`: Framework-agnostic (PyTorch, Jax, NumPy) wrapper classes for common molecular Boltzmann densities and potentially other toy systems (e.g., Lennard Jones, GMM40, ...)

- `boltzkit.utils`: Data augmentation, chirality filtering, as well as general utility functions (e.g., decorators to make functions framework agnostic, pdf-utils, hugging-face wrapper, ...)
    - `.molecular`: Sub-package of `boltzkit.utils`; Provides OpenMM-based utility (Molecular system loading, creation, and energy-, and density-evaluation)
--- 
- `tools`: General ready-to-use tools based on the boltzkit package (e.g., MD dataset creation (for molecular systems), dataset-downloader, evaluation-tool based on file inputs, ...). It is not part of the boltzkit package because it contains ready-to-use applications. 

# Evaluation
The evaluation is modularized. Input is an `EvalData` object with all-optional fields.

**Naming convention:** 
- ground-truth (`true`) samples and predicted (`pred`) samples; must have shape `(batch, dim)`.
- ground-truth density evaluations (`target_log_prob`), which **may be unnormalized**; must have shape `(batch,)`.
- model-density evaluations (`model_log_prob`), which **must be normalized**; must have shape `(batch,)`

**Returns:**
- The returned `metrics` object is a flat `dict` that maps strings to values. Values may be simple `float` or `int` types, but **can also be complex data objects** such as histograms or pdfs. Utilities for filtering and compatibility with Weights and Biases are also available.

### Standard metrics
To run the full evaluation pipeline with general domain-independent metrics, run:
```python
from boltzkit.evaluation import run_eval, EvalData, EnergyHistEval

# Prepare data
data = EvalData(...)
eval_pipeline = [EnergyHistEval()]

metrics = run_eval(data, evals=eval_pipeline)
```

### Custom metrics
To incorporate custom metrics, such as Ramachandran plots for torsion angle marginals in molecular tasks, you can pass one or multiple `Evaluation` objects.
For example, to additionally include torsion marginals, use:
```python
from boltzkit.evaluation import run_eval
from boltzkit.evaluation.molecular_eval import TorsionMarginalEval

# topology can for example be obtained via our `MolecularBoltzmann` target (available under `boltzkit.targets`).
topology = ...
molecular_eval = TorsionMarginalEval(topology) # also allows configuration of metrics via flags

eval_pipeline = [molecular_eval]

metrics = run_eval(
    data,
    evals=eval_pipeline
)
```

### Additional info
By default, an `Evaluation` is skipped with a warning if the provided `data` object does not contain all required fields.
This behavior can be disabled by setting:
```python
metrics = run_eval(data, skip_on_missing_data=False)
```
When `skip_on_missing_data` is set to `False`, a `ValueError` is raised if any required fields are missing for one of the `Evaluation` modules.


## Log to Weights & Biases
**Note**: Requires wandb to be installed!
```python
from boltzkit.evaluation import make_wandb_compatible
import wandb 

# transforms all metrics into wandb-compatible ones
# e.g., converts pdfs into low-resolution images.
# Drops raw data like histograms to not clutter the wandb server.
wandb_metrics = make_wandb_compatible(metrics) 
wandb.log(wandb_metrics)
```

## Save PDFs (for high quality visualizations)
```python
from boltzkit.evaluation import get_pdfs
from boltzkit.utils.pdf import save_pdfs

dir_path = "<some pre-existing directory path>"
pdfs = get_pdfs(metrics)
save_pdfs(pdfs, dir_path) # uses keys of dict as filenames
```

## Save histogram raw data (e.g., for potential custom downstram visualizations)
```python
from boltzkit.evaluation import get_histograms
from boltzkit.utils.histogram import save_histograms

dir_path = "<some pre-existing directory path>"
hists = get_histograms(metrics)
save_histograms(hists, dir_path) # uses keys of dict as filenames
```

# Create target systems
## Boltzmann targets
Boltzmann targets use huggingface and can be instantiated easily via:
```python 
from boltzkit.targets.boltzmann import MolecularBoltzmann

target = MolecularBoltzmann("datasets/chrklitz99/test_system")
```

Some common operations on this target include:
```python
val_samples = target.load_dataset(T=300.0, type="val")
topology = target.get_mdtraj_topology()
tica_model = target.get_tica_model()

target.get_log_prob(samples)
target.get_log_prob_and_score(samples)
target.get_score(samples)
```

### Available Boltzmann targets
- datasets/chrklitz99/test_system
- TODO: missing alanine  dipeptide (test_system is basically that)
- TODO: datasets/chrklitz99/alanine_tetrapeptide
- TODO: missing alanine hexapeptide
- TODO: ELIL tetrapeptide

# Testing
Before running tests, extra packages must be installed with:
```bash
pip install -r requirements_test.txt
```
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


# TODOs
- Chirality filtering
- Test data augmentation