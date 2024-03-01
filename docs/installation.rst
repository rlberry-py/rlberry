.. _rlberry: https://github.com/rlberry-py/rlberry

.. _installation:


Installation
============

First, we suggest you to create a virtual environment using
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

.. code:: bash

    $ conda create -n rlberry
    $ conda activate rlberry


Latest version (0.7.1)
-------------------------------------

Install the latest version for a stable release.

.. code:: bash

    $ pip install rlberry


Development version
--------------------

Install the development version to test new features.

.. code:: bash

    $ pip install rlberry@git+https://github.com/rlberry-py/rlberry.git

.. warning::
    For `zsh` users, `zsh` uses brackets for globbing, therefore it is necessary to add quotes around the argument, e.g. :code:`pip install 'rlberry@git+https://github.com/rlberry-py/rlberry.git'`.


Previous versions
-----------------

If you used a previous version in your work, you can install it by running

.. code:: bash

    $ pip install rlberry@git+https://github.com/rlberry-py/rlberry.git@{TAG_NAME}

replacing `{TAG_NAME}` by the tag of the corresponding version,
e.g., :code:`pip install rlberry@git+https://github.com/rlberry-py/rlberry.git@v0.1`
to install version 0.1.

.. warning::
    For `zsh` users, `zsh` uses brackets for globbing, therefore it is necessary to add quotes around the argument, e.g. :code:`pip install 'rlberry@git+https://github.com/rlberry-py/rlberry.git@v0.1'`.


Deep RL agents
--------------

Deep RL agents require extra libraries, like PyTorch.

* PyTorch agents:

.. code:: bash

    $ pip install rlberry[torch]@git+https://github.com/rlberry-py/rlberry.git

.. warning::
    For `zsh` users, `zsh` uses brackets for globbing, therefore it is necessary to add quotes around the argument, e.g. :code:`pip install 'rlberry[torch_agents]@git+https://github.com/rlberry-py/rlberry.git'`.


Others
-------------
The options for ``pip install rlberry[option]`` are :

- `torch` to install `opencv-python, ale-py, stable-baselines3, tensorboard, torch`
- `extras` to install `optuna, numba, ffmpeg-python, pyvirtualdisplay, scikit-fda, scikit-learn`

(for dev)

- `doc` to install `sphinx, sphinx-gallery, sphinx-math-dollar, numpydoc, myst-parser, sphinxcontrib-video, matplotlib`
