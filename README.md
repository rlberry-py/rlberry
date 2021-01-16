<!-- Logo -->
<p align="center">
   <img src="https://raw.githubusercontent.com/rlberry-py/rlberry/main/assets/logo_wide.svg" width="50%">
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
   <a href='https://rlberry.readthedocs.io/en/latest/?badge=latest'>
      <img alt="Documentation Status" src="https://readthedocs.org/projects/rlberry/badge/?version=latest">
   </a>
   <a href="https://img.shields.io/github/contributors/rlberry-py/rlberry">
      <img alt="contributors" src="https://img.shields.io/github/contributors/rlberry-py/rlberry">
   </a>
   <a href="https://app.codacy.com/gh/rlberry-py/rlberry?utm_source=github.com&utm_medium=referral&utm_content=rlberry-py/rlberry&utm_campaign=Badge_Grade">
      <img alt="Codacy" src="https://api.codacy.com/project/badge/Grade/27e91674d18a4ac49edf91c339af1502">
   </a>
   <a href="https://codecov.io/gh/rlberry-py/rlberry">
      <img alt="codecov" src="https://codecov.io/gh/rlberry-py/rlberry/branch/main/graph/badge.svg?token=TIFP7RUD75">
   </a>
   <a href="https://pypi.org/project/rlberry/">
      <img alt="PyPI" src="https://img.shields.io/pypi/v/rlberry">
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



## What is `rlberry`? 

**Writing reinforcement learning algorithms is fun!** *But after the fun, we have lots of boring things to implement*: run our agents in parallel, average and plot results, optimize hyperparameters, compare to baselines, create tricky environments etc etc!

`rlberry` **is a Python library that makes your life easier** by doing all these things with a few lines of code, so that you can spend most of your time developing agents.

Check our [documentation](https://rlberry.readthedocs.io/en/latest/) or our [getting started section](#getting-started) to see how!


## Contents

| Section | Description |
|-|-|
| [Getting started](#getting-started) | A quick usage guide of `rlberry` |
| [Citation](#citing-rlberry) | How to cite this work |
| [Installation](#installation) | How to install `rlberry` |
| [Contributing](#contributing) | A guide for contributing |


## Getting started

We provide a handful of notebooks on [Google colab](https://colab.research.google.com/) as examples to show you how to use `rlberry`.

| Content | Description | Link |
|-|-|-|
| Introduction to `rlberry` | How to create an agent, optimize its hyperparameters and compare to a baseline. | <a href="https://colab.research.google.com/github/rlberry-py/rlberry/blob/main/notebooks/introduction_to_rlberry.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> |
| RL Experimental Pipeline | How to define a configuration, run experiments in parallel and save a `config.json` for reproducibility. | <a href="https://colab.research.google.com/github/rlberry-py/rlberry/blob/main/notebooks/experimental_pipeline_with_rlberry.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> |


### Compatibility with [OpenAI Gym](https://gym.openai.com/)

If you want to use `gym` environments with `rlberry`, simply do the following:

```python
from rlberry.envs import gym_make

# for example, let's take CartPole
env = gym_make('CartPole-v1')
```

This way, `env` behaves exactly the same as the `gym` environment, we simply replace the seeding function by `env.reseed()`, which ensures unified seeding and reproducibility when using `rlberry`.


### Seeding 

In `rlberry`, __only one global seed__ is defined, and all the random number generators used by the agents and environments inherit from this seed, ensuring __reproducibility__  and __independence between the generators__ (see [NumPy SeedSequence](https://numpy.org/doc/stable/reference/random/parallel.html)).

Example:

```python
import rlberry.seeding as seeding

seeding.set_global_seed(seed=123)

# From now on, no more seeds are defined by the user, and all the results are reproducible.
...

# If you need a random number generator (rng), call:
rng = seeding.get_rng()   

# which gives a numpy Generator (https://numpy.org/doc/stable/reference/random/generator.html) 
# that is independent of all the previous generators created by seeding.get_rng()
rng.integers(5)
rng.normal()
# etc

```

## Citing rlberry

If you use `rlberry` in scientific publications, we would appreciate citations using the following Bibtex entry:

```bibtex
@misc{rlberry,
author = {Domingues, Omar Darwiche and ‪Flet-Berliac, Yannis and Leurent, Edouard and M{\'e}nard, Pierre and Shang, Xuedong and Valko, Michal},
title = {{rlberry - A Reinforcement Learning Library for Research and Education}},
year = {2021},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/rlberry-py/rlberry}},
}
```

## Installation

### Cloning & creating virtual environment

It is suggested to create a virtual environment using Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

```bash
git clone https://github.com/rlberry-py/rlberry.git
conda create -n rlberry python=3.7
```

### Basic installation

Install without heavy libraries (e.g. pytorch):

```bash
conda activate rlberry
pip install -e .
```

### Full installation

Install with all features:

```bash
conda activate rlberry
pip install -e .[full]
```

which includes:

*   [`Numba`](https://github.com/numba/numba) for just-in-time compilation of algorithms based on dynamic programming
*   [`PyTorch`](https://pytorch.org/) for Deep RL agents
*   [`Optuna`](https://optuna.org/#installation) for hyperparameter optimization
*   [`ffmpeg-python`](https://github.com/kkroening/ffmpeg-python) for saving videos
*   [`PyOpenGL`](https://pypi.org/project/PyOpenGL/) for more rendering options

### Tests

To run tests, install test dependencies with `pip install -e .[test]` and run `pytest`. To run tests with coverage, install test dependencies and run `bash run_testscov.sh`. See coverage report in `cov_html/index.html`.

## Contributing

Want to contribute to `rlberry`? Please check [our contribution guidelines](CONTRIBUTING.md). A list of interesting TODO's will be available soon. **If you want to add any new agents or environments, do not hesitate to [open an issue](https://github.com/rlberry-py/rlberry/issues/new/choose)!**

### Implementation notes

*   When inheriting from the `Agent` class, make sure to call `Agent.__init__(self, env, **kwargs)` using `**kwargs` in case new features are added to the base class, and to make sure that `copy_env` and `reseed_env` are always an option to any agent. 

Infos, errors and warnings are printed using the `logging` library.

*   From `gym` to `rlberry`:
    *   `reseed` (rlberry) should be called instead of `seed` (gym). `seed` keeps compatilibity with gym, whereas `reseed` uses the unified seeding mechanism of `rlberry`.
