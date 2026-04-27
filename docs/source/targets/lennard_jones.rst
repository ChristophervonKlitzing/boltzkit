Lennard Jones
=====================

Overview
--------

The ``LennardJones`` class defines a Lennard Jones potential with choosable number of particles and spatial dimensions.

Instantiation
-------------

A Lennard Jones model can be created easily:

.. code-block:: python

   import numpy as np
   from boltzkit.targets import LennardJones

   target = LennardJones(n_particles=13)

