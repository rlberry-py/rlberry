.. _rlberry: https://github.com/rlberry-py/rlberry

.. _installation:


Installation
============

First, we suggest you to create a virtual environment using 
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

.. code:: bash

    $ conda create -n rlberry python=3.7
    $ conda activate rlberry

Then you have two options for the installation. For a stable version, you can just install by:

.. code:: bash

    $ pip install rlberry[default]


For more advanced users who want to try the development version, all you need to do is clone the rlberry_ repository and install:

.. code:: bash

    $ git clone https://github.com/rlberry-py/rlberry.git
    $ cd rlberry
    $ pip install -e .[default]


Installation for Deep RL agents
===============================

Deep RL agents require extra libraries, like PyTorch and JAX.

* PyTorch agents:

.. code:: bash
    $ pip install -e .[torch_agents]
    $ pip install tensorboard   # only if you're not installing jax_agents too!
* JAX agents:

.. code:: bash
    $ pip install -e .[jax_agents]

.. warning::
    If you're using PyTorch agents *and* JAX agents, do not install tensorboard separately,
    since `pip install -e .[jax_agents]` installs tensorflow, which already contains
    tensorboard. Otherwise, there might be a conflict between the two installations
    and tensorboard will not work properly.
