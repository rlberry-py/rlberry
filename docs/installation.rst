.. _rlberry: https://github.com/rlberry-py/rlberry

.. _installation:


Installation
============

First, we suggest you to create a virtual environment using 
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

Then, all you need to do is clone the rlberry_ repository and install:

.. code:: bash

    $ conda create -n rlberry python=3.7
    $ conda activate rlberry
    $ git clone https://github.com/rlberry-py/rlberry.git
    $ cd rlberry
    $ pip install -e .[full]

or, for a basic installation (without heavy libraries like pytorch):

.. code:: bash

    $ pip install -e .

Full installation includes, for instance:

*   `Numba <https://github.com/numba/numba>`_ for just-in-time compilation of algorithms based on dynamic programming
*   `PyTorch <https://pytorch.org/>`_ for Deep RL agents
*   `Optuna <https://optuna.org/#installation>`_ for hyperparameter optimization
*   `ffmpeg-python <https://github.com/kkroening/ffmpeg-python>`_ for saving videos
*   `PyOpenGL <https://pypi.org/project/PyOpenGL/>`_ for more rendering options

