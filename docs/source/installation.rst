Installation
============




Prerequisites
--------------------

The ``boltzkit`` package requires ``python>=3.10`` with OpenMM being installed separately.


Install OpenMM
~~~~~~~~~~~~~~

OpenMM can be installed via either pip or conda, and it may use either CPU or GPU (CUDA) backends depending on your system. For this reason, OpenMM is typically installed separately.

CPU installation (pip)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install openmm


GPU installation (conda)
^^^^^^^^^^^^^^^^^^^^^^^^^

For example, to install OpenMM with CUDA 12 support using conda:

.. code-block:: bash

   conda install -c conda-forge openmm cuda-version=12





Installing boltzkit
-------------------

The latest version of ``boltzkit`` can be installed via:

.. code-block:: bash

   pip install git+https://github.com/ChristophervonKlitzing/boltzkit






Development setup
-------------------

Clone and install in editable mode with development dependencies:

.. code-block:: bash

   git clone git@github.com:ChristophervonKlitzing/boltzkit.git


Inside the cloned package, all dependencies can be installed using the extra ``-e`` flag for an editable install and ``[dev]`` flag for the extra development dependencies:

.. code-block:: bash

   pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cpu