.. _rlberry: https://github.com/rlberry-py/rlberry

.. _installation:


Installation
============

First, we suggest you to create a virtual environment using 
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

.. code:: bash

    $ conda create -n rlberry python=3.8
    $ conda activate rlberry

**Installing the latest version (0.2.1):**

.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git@v0.2.1#egg=rlberry[default]

Alternatively, you can clone the repository and run :code:`pip install -e .[default]`.


**Installing the development version:**

.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git@dev#egg=rlberry[default]


**Installing a previous version:**


.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git@{TAG_NAME}#egg=rlberry[default]

replacing `{TAG_NAME}` by the tag of the corresponding version,
e.g., :code:`pip install git+https://github.com/rlberry-py/rlberry.git@v0.1#egg=rlberry[default]`
to install version 0.1.

.. warning::
    For `zsh` users, `zsh` uses brackets for globbing, therefore it is necessary to add quotes around the argument, e.g. :code:`pip install 'git+https://github.com/rlberry-py/rlberry.git#egg=rlberry[default]'`.


Installation for Deep RL agents
===============================

Deep RL agents require extra libraries, like PyTorch and JAX.

* PyTorch agents:

.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git#egg=rlberry[torch_agents]
    $ pip install tensorboard   # only if you're not installing jax_agents too!

* JAX agents (**Linux only**):

.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git#egg=rlberry[jax_agents]

.. warning::
    If you're using PyTorch agents *and* JAX agents, do not install tensorboard separately,
    since :code:`pip install -e .[jax_agents]` installs tensorflow, which already contains
    tensorboard. Otherwise, there might be a conflict between the two installations
    and tensorboard will not work properly.
