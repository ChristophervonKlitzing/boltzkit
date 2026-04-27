Mixture of Gaussians
=====================

Overview
--------

The :class:`boltzkit.targets.gaussian_mixture.DiagonalGaussianMixture` class defines a Gaussian mixture model (GMM) with **diagonal covariance matrices**.

Each component is a multivariate normal distribution with independent dimensions:

.. math::

   \Sigma_k = \mathrm{diag}(\sigma_{k,1}^2, \ldots, \sigma_{k,D}^2)

The mixture weights are parameterized via logits and normalized internally using a log-softmax.

Instantiation
-------------

A mixture model can be created directly from parameters:

.. code-block:: python

   import numpy as np
   from boltzkit.targets import DiagonalGaussianMixture

   means = np.array([
       [-2.0, -2.0],
       [ 2.0,  2.0],
   ])

   diag_stds = np.array([
       [0.8, 0.8],
       [0.5, 0.9],
   ])

   logits = np.array([0.4, 0.6])

   target = DiagonalGaussianMixture(means, diag_stds, logits)


Structure
----------

- ``means``: shape ``(K, D)``
- ``diag_stds``: shape ``(K, D)``, strictly positive
- ``logits``: shape ``(K,)`` (normalized internally)


Factory constructors
--------------------

Isotropic mixture
~~~~~~~~~~~~~~~~~~

All components share the same standard deviation:

.. code-block:: python

   target = DiagonalGaussianMixture.create_isotropic(
       means=means,
       std=1.0,
       logits=logits
   )

Uniform isotropic mixture (e.g., GMM40)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Means are sampled uniformly in a range and weights are equal:

.. code-block:: python

   target = DiagonalGaussianMixture.create_isotropic_uniform(
       std=1.0,
       n_components=40,
       dim=2,
       mean_range=(-40.0, 40.0),
       seed=0
   )

Predefined GMM (toy benchmark)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A standard test configuration:

.. code-block:: python

   target = DiagonalGaussianMixture.create_gmm40()


Sampling
--------

Samples can be drawn directly from the mixture:

.. code-block:: python

   samples = target.sample(n_samples=1000, seed=0)


Dataset generation
-------------------

A deterministic synthetic dataset can be generated via:

.. code-block:: python

   dataset = target.load_dataset(
       type="val",
       length=1000,
       # + optional arguments to automatically include log_probs and/or scores
   )
   

Properties:

- Deterministic for fixed ``(type, seed)``, defines the entire infinite sequence of samples
- Reproducible across calls
- Supports ``train``, ``val``, ``test`` splits
- Returns the first ``length`` samples of the sequence; increasing ``length`` preserves the existing prefix and appends additional samples without actually caching anything using procedural generation.

Interpretation
--------------

The model defines:

.. math::

   p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)

where:

- :math:`\pi_k` are normalized mixture weights
- :math:`\Sigma_k` are diagonal covariance matrices
