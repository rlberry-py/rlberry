.. _rlberry: https://github.com/rlberry-py/rlberry

.. _installation:


Installation
============

First, we suggest you to create a virtual environment using
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

.. code:: bash

    $ conda create -n rlberry
    $ conda activate rlberry

OS dependency
-------------

In order to render videos in rlberry, `ffmpeg <https://ffmpeg.org/>`_ must be installed.

Latest version (0.5.0)
-------------------------------------

Install the latest version for a stable release.

.. code:: bash

    $ pip install rlberry


Development version
--------------------

Install the development version to test new features.

.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git#egg=rlberry[default]

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

Deep RL agents require extra libraries, like PyTorch.

* PyTorch agents:

.. code:: bash

    $ pip install git+https://github.com/rlberry-py/rlberry.git#egg=rlberry[torch_agents]
    $ pip install tensorboard



* Stable-baselines3 agents with Gymnasium support:
    (https://github.com/DLR-RM/stable-baselines3/pull/1327)
.. code:: bash

    $ pip install git+https://github.com/DLR-RM/stable-baselines3@feat/gymnasium-support
    $ pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/gymnasium-support
