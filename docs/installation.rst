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

    $ pip install rlberry[full]

or, for a basic installation (without heavy libraries like PyTorch):

.. code:: bash

    $ pip install rlberry

For more advanced users who want to try the development version, all you need to do is clone the rlberry_ repository and install:

.. code:: bash

    $ git clone https://github.com/rlberry-py/rlberry.git
    $ cd rlberry
    $ pip install -e .[full]

or, for a basic installation:

.. code:: bash

    $ pip install -e .

Full installation includes, for instance:

*   `Numba <https://github.com/numba/numba>`_ for just-in-time compilation of algorithms based on dynamic programming
*   `PyTorch <https://pytorch.org/>`_ for Deep RL agents
*   `Optuna <https://optuna.org/#installation>`_ for hyperparameter optimization
*   `ffmpeg-python <https://github.com/kkroening/ffmpeg-python>`_ for saving videos
*   `PyOpenGL <https://pypi.org/project/PyOpenGL/>`_ for more rendering options

