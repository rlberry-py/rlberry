<!-- Logo -->
<p align="center">
   <img src="logo/logo_wide.svg" width="50%">
</p>

<!-- Short description -->
<p align="center">
   A Reinforcement Learning Library for Research and Education
</p>

<!-- The badges -->
<p align="center">
   <a href="https://github.com/rlberry-py/rlberry/workflows/test/badge.svg">
      <img alt="pytest" src="https://github.com/rlberry-py/rlberry/workflows/test/badge.svg">
   </a>
   <a href="https://img.shields.io/github/contributors/rlberry-py/rlberry">
      <img alt="contributors" src="https://img.shields.io/github/contributors/rlberry-py/rlberry">
   </a>
   <a href="https://app.codacy.com/gh/rlberry-py/rlberry?utm_source=github.com&utm_medium=referral&utm_content=rlberry-py/rlberry&utm_campaign=Badge_Grade">
      <img alt="Codacy" src="https://api.codacy.com/project/badge/Grade/27e91674d18a4ac49edf91c339af1502">
   </a>
   <a href="https://codecov.io/gh/rlberry-py/rlberry">
      <img src="https://codecov.io/gh/rlberry-py/rlberry/branch/main/graph/badge.svg?token=TIFP7RUD75"/>
   </a> 
</p>

<p align="center">
   <a href="https://colab.research.google.com/github/rlberry-py/rlberry/blob/main/notebooks/introduction_to_rlberry.ipynb">
      <b>Try it on Google Colab!</b>
      <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
   </a>
</p>

<!-- Horizontal rule -->
<hr>

<!-- Table of content -->

| Section | Description |
|-|-|
| [Goals](#goals) | The philosophy of the library |
| [Installation](#installation) | How to install the library |
| [Getting started](#getting-started) | A quick guide on how to use rlberry |
| [Documentation](#documentation) | A link to the documentation |
| [Contributing](#contributing) | A guide for contributing |
| [Citation](#citing-rlberry) | How to cite this work |

## Goals

*   Write detailed documentation and comprehensible tutorial/examples (Jupyter Notebook) for each implemented algorithm.

*   Provide a general interface for agents, that
    *   puts minimal constraints on the agent code (=> making it easy to include new algorithms and modify existing ones);

    *   allows comparison between agents using a simple and unified evaluation interface (=> making it easy, for instance, to compare deep and "traditional" RL algorithms).
    
*   Unified seeding mechanism: define only one global seed, from which all other seeds will inherit, enforcing independence of the random number generators.

*   Simple interface for creating and rendering new environments. 

## Installation

### Cloning & creating virtual environment

It is suggested to create a virtual environment using Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

```bash
git clone https://github.com/rlberry-py/rlberry.git
conda create -n rlberry python=3.7
```

### Basic installation

Install without heavy libraries (e.g. pytorch).

```bash
conda activate rlberry
pip install -e .
```

### Full installation

Install with all features,

```bash
conda activate rlberry
pip install -e .[full]
```

which includes:

*   [`Numba`](https://github.com/numba/numba) for just-in-time compilation of algorithms based on dynamic programming,
*   [`PyTorch`](https://pytorch.org/) for Deep RL agents,
*   [`Optuna`](https://optuna.org/#installation) for hyperparameter optimization,
*   [`ffmpeg-python`](https://github.com/kkroening/ffmpeg-python) for saving videos,
*   [`PyOpenGL`](https://pypi.org/project/PyOpenGL/) for more rendering options.

## Getting started

### Tests

To run tests, run `pytest`. To run tests with coverage, install and run `pytest-cov`:

```shell
pip install pytest-cov
bash run_tests.sh
```

See coverage report in `cov_html/index.html`.

## Documentation

## Contributing

### Implementation notes

*   When inheriting from the `Agent` class, make sure to call `Agent.__init__(self, env, **kwargs)` using `**kwargs` in case new features are added to the base class, and to make sure that `copy_env` and `reseed_env` are always an option to any agent. 

*   Convention for verbose in the agents:
    *   `verbose=0`: nothing is printed
    *   `verbose>1`: print progress messages

Errors and warnings are printed using the `logging` library.

## Citing rlberry
