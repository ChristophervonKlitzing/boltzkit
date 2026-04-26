Evaluation
==========

Overview
--------

The evaluation system is modular and operates on an ``EvalData`` object. All fields in ``EvalData`` are optional and only required depending on the selected evaluation modules.

Evaluation results are returned as a flat ``metrics`` dictionary. Depending on the selected modules, values may include scalars, histograms, or binary artifacts such as PDF files.

Data conventions
----------------

- Ground-truth samples (``true``) and predicted samples (``pred``) have shape ``(batch, dim)``.
- Ground-truth density evaluations (``target_log_prob``) may be unnormalized and have shape ``(batch,)``.
- Model density evaluations (``model_log_prob``) must be normalized and have shape ``(batch,)``.

Building and running an evaluation pipeline
-------------------------------

Evaluation is performed by composing a list of modular evaluators.

Example
~~~~~~~

.. code-block:: python

   from boltzkit.evaluation import run_eval, EvalData, EnergyHistEval
   from boltzkit.evaluation.molecular_eval import TorsionMarginalEval

   data = EvalData(...)

   topology = ...
   torsion_eval = TorsionMarginalEval(topology)
   energy_eval = EnergyHistEval()

   eval_pipeline = [energy_eval, torsion_eval]

   metrics = run_eval(data, evals=eval_pipeline)


Missing data behavior
~~~~~~~~~~~~~~~~~~~~~~~~~

By default, evaluation modules that cannot be computed due to missing fields in ``EvalData`` are skipped with a warning.

To enforce strict execution (raise an error instead), disable skipping:

.. code-block:: python

   metrics = run_eval(data, skip_on_missing_data=False)

In this mode, a ``ValueError`` is raised if any required inputs for an evaluation module are missing.

Output format
~~~~~~~~~~~~~~~~~~~~~~~~~

The returned ``metrics`` object is a flat dictionary:

- Keys: metric names (strings)
- Values: can be
  - Scalars (``float``, ``int``)
  - Structured data (e.g., histograms)
  - Binary artifacts (e.g., PDF files stored as byte buffers)

Utility functions are available to extract or filter subsets of metrics depending on downstream use.

Logging and export
------------------

Weights & Biases logging
~~~~~~~~~~~~~~~~~~~~~~~~~

``wandb`` is optional and must be installed separately.

To log evaluation results:

.. code-block:: python

   from boltzkit.evaluation import make_wandb_compatible
   import wandb
   
   # transforms all metrics into wandb-compatible ones
   # e.g., converts pdfs into low-resolution images.
   # Drops raw data like histograms to not clutter the wandb server.
   wandb_metrics = make_wandb_compatible(metrics)
   wandb.log(wandb_metrics)


PDF artifacts (binary buffers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some evaluations produce visualizations as PDF files stored in-memory as binary buffers. These can be written directly to disk:

.. code-block:: python

    from boltzkit.evaluation import get_pdfs
    from boltzkit.utils.pdf import save_pdfs

    dir_path = "<some pre-existing directory path>"
    pdfs = get_pdfs(metrics)
    save_pdfs(pdfs, dir_path) # uses keys of dict as filenames



Histograms
~~~~~~~~~~~~~~~~~~~~~~

The density-counts of histograms can be exported for custom downstream analysis (e.g., visualizations):

.. code-block:: python

   from boltzkit.evaluation import get_histograms
   from boltzkit.utils.histogram import save_histograms

   dir_path = "<existing directory>"
   hists = get_histograms(metrics)
   save_histograms(hists, dir_path) # uses keys of dict as filenames