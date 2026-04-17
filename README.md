![Tests (main)](https://img.shields.io/github/actions/workflow/status/ChristophervonKlitzing/boltzkit/run-tests.yaml?branch=main&label=tests%20(main))
![Tests (dev)](https://img.shields.io/github/actions/workflow/status/ChristophervonKlitzing/boltzkit/run-tests.yaml?branch=dev&label=tests%20(dev))

# The boltzkit package
`boltzkit` is a Python package for working with molecular Boltzmann and related densities.
It provides tools for defining target systems, loading datasets, running evaluation pipelines, and performing tasks such as chirality filtering.

The package is designed to make it as simple as possible to:
- evaluate target energies / log probabilities
- compute forces / scores
- evaluate predicted samples

See the [Evaluation](#create-and-run-evaluation-pipeline) and 
[Create Target Systems](#create-target-systems) sections.


The `boltzkit` package is **framework-agnostic**! The package relies primarily on NumPy and deliberately avoids direct dependencies on machine learning frameworks such as JAX or PyTorch. Optional wrappers are provided for these frameworks when needed.

This design ensures maximum flexibility and broad compatibility with any NumPy-based or NumPy-compatible machine learning framework.


# Installation
## Required prior setup
The boltzkit package requires python>=3.10 with OpenMM being installed separately.

### Create conda environment
To create a fresh python environment use
```bash
conda create -n some_fancy_env_name python=3.10 pip
```


### Install OpenMM
OpenMM can be installed via either pip or conda, and it may use either CPU or GPU (CUDA) backends depending on your system. For this reason, OpenMM is typically installed separately.

#### CPU installation (pip)
```bash
pip install openmm
```

#### GPU installation (conda)
For example, to install OpenMM with CUDA 12 support using conda:
```bash
conda install -c conda-forge openmm cuda-version=12
```

## Installing boltzkit
The `boltzkit` package can either be installed as a normal pip package or for development on itself.

### For regular use
The latest version of `boltzkit` can directly be installed from GitHub via:
```bash
pip install git+https://github.com/ChristophervonKlitzing/boltzkit
```

### Development setup
Clone the repository using
```bash
git clone git@github.com:ChristophervonKlitzing/boltzkit.git
```

Inside the cloned package, all dependencies can be installed using the extra `-e` flag for an editable install and `[dev]` flag for the extra development dependencies:
```bash
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cpu
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

### Create and run evaluation pipeline
To run an evaluation pipeline with energy histograms and torsion marginal metrics (e.g., Ramachandrans), run:
```python
from boltzkit.evaluation import run_eval, EvalData, EnergyHistEval
from boltzkit.evaluation.molecular_eval import TorsionMarginalEval

# Prepare data
data = EvalData(...)


# mdtraj.Topology can for example be obtained via our `MolecularBoltzmann` target (available under `boltzkit.targets`) or by loading a PDB file with mdtraj.
topology = ...
torsion_eval = TorsionMarginalEval(topology) # allows additional configuration of metrics via flags
energy_hist_eval = EnergyHistEval()

# Create pipeline
eval_pipeline = [energy_hist_eval, torsion_eval]

# Run evaluation
metrics = run_eval(data, evals=eval_pipeline)
```

### Additional info
By default, an `Evaluation` is skipped with a warning if the provided `data` object does not contain all required fields.
This behavior can be disabled by setting:
```python
metrics = run_eval(data, skip_on_missing_data=False)
```
When `skip_on_missing_data` is set to `False`, a `ValueError` is raised if any required fields are missing for one of the `Evaluation` modules.


## Log to Weights & Biases
**Note**: Requires wandb to be installed (not a dependency of boltzkit)!
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

# Target systems
## Framework support
Each target implements `get_log_prob`, `get_score`, and `get_log_prob_and_score`, which accept inputs as NumPy arrays, JAX arrays, or PyTorch tensors and return outputs in the corresponding framework format. The table below summarizes framework support for each target (supported: ✅ | not supported: ❌).
| Target                        | NumPy | Jax | PyTorch |
|------------------------------|:-----:|:---:|:-------:|
| MolecularBoltzmann          |   ✅   |  ✅  |    ✅    |
| DiagonalGaussianMixture     |   ✅   |  ✅  |    ✅    |
| IsotropicGaussianMixture    |   ✅   |  ✅  |    ✅    |

## Boltzmann targets
Boltzmann targets use huggingface and can be instantiated easily via:
```python 
from boltzkit.targets.boltzmann import MolecularBoltzmann

target = MolecularBoltzmann("datasets/chrklitz99/alanine_dipeptide")
```

Some common operations on this target include:
```python
val_dataset = target.load_dataset(T=300.0, type="val")
topology = target.get_mdtraj_topology()
tica_model = target.get_tica_model()

target.get_log_prob(samples)
target.get_log_prob_and_score(samples)
target.get_score(samples)
```

### Available Boltzmann targets
- datasets/chrklitz99/alanine_dipeptide 
- datasets/chrklitz99/alanine_tetrapeptide
- datasets/chrklitz99/alanine_hexapeptide

### Change length scale (Ångström, Nanometer, ...)

Atomic coordinates (e.g., samples, scores, or forces) can be scaled arbitrarily. Internally, `MolecularBoltzmann` represents all coordinates in **nanometers**. The length scale can be configured when creating a target:

```python
target = MolecularBoltzmann(
    "datasets/chrklitz99/alanine_dipeptide",
    length_unit="angstrom"  # default: "nanometer"
)
```

**Note:** When setting the length scale during target creation, the `target` API automatically adapts. All inputs and outputs (coordinates, forces, scores, etc.) of the public class API will use the selected unit, making it easy to work in the selected unit.

For this reason, we strongly recommend running the [evaluation](#create-and-run-evaluation-pipeline) together with `MolecularBoltzmann` targets. Otherwise, it is the user's responsibility to ensure that a consistent length scale is used throughout the evaluation.

Alternatively, a positive real number can be passed to specify the coordinate scaling directly. For example:

- `"nanometer"` corresponds to `length_unit = 1.0`
- `"angstrom"` corresponds to `length_unit = 0.1`



# Testing
Ensure the dev requirements are installed (see [development setup](#development-setup)).
## Run unit tests
To run all unittests, inside the boltzkit repo run:
```bash
python -m unittest
```
Alternatively, specific unittests can be executed with:
```bash
python -m unittest tests.demo
```

## Create unit test
test files need to start with `test_` to be discoverable.




# Molecular dynamics trajectories

**Note:** This section only applies when generating new datasets is necessary.  
For the systems already included in this repository, the corresponding trajectories have mostly already been generated for 300K. This chapter requires the `dev` requirements to be installed (see [development setup](#development-setup)).

Equilibrium-distribution trajectories can be generated using the `tools/run_simulation.py` script. For each system, we generate **two independent trajectories**, each containing **10⁷ samples**.

- **Trajectory 1** is used directly as the **test dataset** and for **training the TICA model**. The **test dataset** is a random permutation of this trajectory.
- **Trajectory 2** is **subsampled without replacement** to construct the **training** and **validation** datasets, each containing **10⁶ samples**.


### Commands
The two trajectories each of size 10⁷ for the system **Alanine Dipeptide** were generated by running the following command twice: 
```bash
python tools/run_simulation.py --system datasets/chrklitz99/alanine_dipeptide --temps 300.0 --time_step 1.0 --rec_interval 0.5 --pre_eq_time 200.0 --simu_time 5000.0 --integrator LangevinMiddle --write_checkpoint_every_ns 100
```
---
The four trajectories each of size 5x10⁶ for the system **Alanine Tetrapeptide** were generated by running the following command four times:

```bash
python tools/run_simulation.py --system datasets/chrklitz99/alanine_tetrapeptide --temps 300.0:500.0:6 --time_step 1.0 --rec_interval 0.2 --pre_eq_time 200.0 --simu_time 1200.0 --integrator LangevinMiddle --write_checkpoint_every_ns 100 --save_traj_of_replicas 0
```
---

The four trajectories each of size 5x10⁶ for the system **Alanine Hexapeptide** were generated by running the following command four times:
```bash
python tools/run_simulation.py --system datasets/chrklitz99/alanine_hexapeptide --temps 300.0:500.0:6 --time_step 1.0 --rec_interval 0.2 --pre_eq_time 200.0 --simu_time 1200.0 --integrator LangevinMiddle --write_checkpoint_every_ns 100 --save_traj_of_replicas 0
```



### Workflow to create TICA model and datasets
1. **Convert to NumPy:** Convert the raw `.h5` output to NumPy format. Use the `--skipN` flag for larger systems to discard the initial equilibration phase (e.g., `--skipN 1000000` for Alanine Hexapeptide to remove the first 200ns of REMD).
    ```bash
    python tools/extract_trajectory_as_numpy.py path/to/traj.h5 --skipN <N>
    ```
2. **Create TICA Model:** Run the model creation script on the trajectory designated for the test dataset. This identifies slow degrees of freedom and offers options for plotting lag-times or Ramachandran correspondences.
    ```bash
    python tools/create_tica_model.py --traj_path path/to/traj.npy --traj_total_sim_time_ns <sim_time> --system_name <system_path> --lag_time_ps 100
    ```
3. **Generate Test Dataset:** Permute the first trajectory. If multiple parallel trajectories were generated, concatenate them before running this command:
    ```bash
    python tools/permute_trajectory.py path/to/traj_1.npy
    ```
4. **Generate Train/Val Datasets:** Create random subsets without replacement from the second trajectory:
    ```bash
    python tools/split_trajectory.py path/to/traj_2.npy
    ```
5. Upload everything to huggingface (see alanine dipeptide for reference) and don't forget to update the info.yaml file


# TODOs
- Chirality filtering
- Test data augmentation