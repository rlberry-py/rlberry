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
</p>

<p align="center">
   <!-- <a href="https://img.shields.io/pypi/pyversions/rlberry">
      <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/rlberry">
   </a> -->
</p>

<p align="center">
   <!-- <a href="https://pypi.org/project/rlberry/">
      <img alt="PyPI" src="https://img.shields.io/pypi/v/rlberry">
   </a> -->
   <!-- <a href="https://img.shields.io/pypi/wheel/rlberry">
      <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/rlberry">
   </a> -->
   <!-- <a href="https://img.shields.io/pypi/status/rlberry">
      <img alt="PyPI - Status" src="https://img.shields.io/pypi/status/rlberry">
   </a> -->
   <!-- <a href="https://img.shields.io/pypi/dm/rlberry">
      <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/rlberry">
   </a> -->
   <!-- <a href="https://zenodo.org/badge/latestdoi/304451364">
      <img src="https://zenodo.org/badge/304451364.svg" alt="DOI">
   </a> -->
</p>

<p align="center">
   <a href="https://colab.research.google.com/github/rlberry-py/notebooks/blob/main/introduction_to_rlberry.ipynb">
      <b>Try it on Google Colab!</b>
      <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
   </a>
</p>

<!-- Horizontal rule -->
<hr>

<!-- Table of content -->

## What is `rlberry`?

**Writing reinforcement learning algorithms is fun!** *But after the fun, we have lots of boring things to implement*:
run our agents in parallel, average and plot results, optimize hyperparameters, compare to baselines, create tricky
environments etc etc!

`rlberry` **is a Python library that makes your life easier** by doing all these things with a few lines of code, so
that you can spend most of your time developing agents.
`rlberry` also provides implementations of several RL agents, benchmark environments and many other useful tools.

## Installation

Install the latest version for a stable release.

```bash
pip install -U git+https://github.com/rlberry-py/rlberry.git@v0.3.0#egg=rlberry[default]
```

The documentation includes more [installation instructions](https://rlberry.readthedocs.io/en/latest/installation.html) in particular for users that work with Jax.


## Getting started

In our [documentation](https://rlberry.readthedocs.io/en/latest/), you will find [quick starts](https://rlberry.readthedocs.io/en/latest/user_guide.html#quick-start-setup-an-experiment-and-evaluate-different-agents) to the library and a [user guide](https://rlberry.readthedocs.io/en/latest/user_guide.html) with a few tutorials on using rlberry.

Also, we provide a handful of notebooks on [Google colab](https://colab.research.google.com/) as examples to show you
how to use `rlberry`:

| Content | Description | Link |
|-|-|-|
| Introduction to `rlberry` | How to create an agent, optimize its hyperparameters and compare to a baseline.| <a href="https://colab.research.google.com/github/rlberry-py/notebooks/blob/main/introduction_to_rlberry.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>|
| Evaluating and optimizing agents | Train a REINFORCE agent and optimize its hyperparameters|  <a href="https://colab.research.google.com/github/rlberry-py/notebooks/blob/main/rlberry_evaluate_and_optimize_agent.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

## Changelog

See the [changelog](https://rlberry.readthedocs.io/en/latest/changelog.html) for a history of the chages made to rlberry.

## Citing rlberry

If you use `rlberry` in scientific publications, we would appreciate citations using the following Bibtex entry:

```bibtex
@misc{rlberry,
    author = {Domingues, Omar Darwiche and Flet-Berliac, Yannis and Leurent, Edouard and M{\'e}nard, Pierre and Shang, Xuedong and Valko, Michal},
    doi = {10.5281/zenodo.5544540},
    month = {10},
    title = {{rlberry - A Reinforcement Learning Library for Research and Education}},
    url = {https://github.com/rlberry-py/rlberry},
    year = {2021}
}
```


## Development notes

The modules listed below are experimental at the moment, that is, they are not thoroughly tested and are susceptible to evolve.

* `rlberry.network`: Allows communication between a server and client via sockets, and can be used to run agents remotely.

* `rlberry.agents.torch`, `rlberry.agents.jax`, `rlberry.exploration_tools.torch`: Deep RL agents are currently not stable, and their main purpose now is to illustrate how to implement and run those algorithms with the `rlberry` interface
(e.g., run several agents in parallel, optimize hyperparameters etc.).
Other libraries, such as [Stable Baselines](https://stable-baselines3.readthedocs.io/en/master/) provide reliable implementations of deep RL algorithms, and **can be used with `rlberry`**, as shown by
[this example](https://rlberry.readthedocs.io/en/latest/auto_examples/demo_examples/demo_from_stable_baselines.html#sphx-glr-auto-examples-demo-examples-demo-from-stable-baselines-py).


## Contributing

Want to contribute to `rlberry`? Please check [our contribution guidelines](https://rlberry.readthedocs.io/en/latest/contributing.html). **If you want to add any new agents or environments, do not hesitate
to [open an issue](https://github.com/rlberry-py/rlberry/issues/new/choose)!**
