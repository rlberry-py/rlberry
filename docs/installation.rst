.. _rlberry: https://github.com/rlberry-py/rlberry

.. _installation:


Installation
============

First, we suggest you to create a virtual environment using
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

.. code:: bash

    $ conda create -n rlberry python=3.8
    $ conda activate rlberry


Latest version (0.3.0)
-------------------------------------

Install the latest version for a stable release.

.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git@v0.3.0#egg=rlberry[default]


Development version
--------------------

Install the development version to test new features.

.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git#egg=rlberry[default]

.. warning::

   When using Python 3.10, there seem to be a problem when installing PyOpenGL-accelerate. For
   now, we advise people to use Python 3.9 with PyOpenGL==3.1.5 and PyOpenGL-accelerate==3.1.5.
   It is also possible to use rlberry without installing PyOpenGL-accelerate but this could cause
   rendering to be slow.


Previous versions
-----------------

If you used a previous version in your work, you can install it by running

.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git@{TAG_NAME}#egg=rlberry[default]

replacing `{TAG_NAME}` by the tag of the corresponding version,
e.g., :code:`pip install git+https://github.com/rlberry-py/rlberry.git@v0.1#egg=rlberry[default]`
to install version 0.1.

.. warning::
    For `zsh` users, `zsh` uses brackets for globbing, therefore it is necessary to add quotes around the argument, e.g. :code:`pip install 'git+https://github.com/rlberry-py/rlberry.git#egg=rlberry[default]'`.


Deep RL agents
--------------

Deep RL agents require extra libraries, like PyTorch and JAX.

* PyTorch agents:

.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git#egg=rlberry[torch_agents]
    $ pip install tensorboard   # only if you're not installing jax_agents too!

* JAX agents (**Linux only, experimental**):

.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git#egg=rlberry[jax_agents]

.. warning::
    If you're using PyTorch agents *and* JAX agents, do not install tensorboard separately,
    since :code:`pip install -e .[jax_agents]` installs tensorflow, which already contains
    tensorboard. Otherwise, there might be a conflict between the two installations
    and tensorboard will not work properly.
