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
   <a href="https://pypi.org/project/rlberry/">
      <img alt="Python Version" src="https://img.shields.io/badge/python-3.11-blue">
   </a>
   <a href="https://img.shields.io/github/contributors/rlberry-py/rlberry">
      <img alt="contributors" src="https://img.shields.io/github/contributors/rlberry-py/rlberry">
   </a>
   <a href="https://codecov.io/gh/rlberry-py/rlberry">
      <img alt="codecov" src="https://codecov.io/gh/rlberry-py/rlberry/branch/main/graph/badge.svg?token=TIFP7RUD75">
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

We provide you a number of tools to help you achieve **reproducibility**, **statistically comparisons** of RL agents, and **nice visualization**.

## Installation

Install the latest (minimal) version for a stable release.

```bash
pip install -U git+https://github.com/rlberry-py/rlberry.git@v0.3.0#egg=rlberry[default]
```

The documentation includes more [installation instructions](https://rlberry-py.github.io/rlberry/installation.html).


## Getting started

In our [dev documentation](https://rlberry-py.github.io/rlberry/), you will find [quick starts](https://rlberry-py.github.io/rlberry/basics/quick_start_rl/quickstart.html#quick-start) to the library and a [user guide](https://rlberry-py.github.io/rlberry/user_guide.html) with a few tutorials on using rlberry, and some [examples](https://rlberry-py.github.io/rlberry/auto_examples/index.html). See also the [stable documentation](https://rlberry-py.github.io/rlberry/stable/) for the documentation corresponding to the last release.


## Changelog

See the [changelog](https://rlberry-py.github.io/rlberry/changelog.html) for a history of the chages made to rlberry.

## Other rlberry projects

[rlberry-scool](https://github.com/rlberry-py/rlberry-scool) : It’s the repository used for teaching purposes. These are mainly basic agents and environments, in a version that makes it easier for students to learn.

[rlberry-research](https://github.com/rlberry-py/rlberry-research) : It’s the repository where our research team keeps some agents, environments, or tools compatible with rlberry. It’s a permanent “work in progress” repository, and some code may be not maintained anymore.


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

## About us
This project was initiated and is actively maintained by [INRIA SCOOL team](https://team.inria.fr/scool/).
More information [here](https://rlberry-py.github.io/rlberry/stable/about.html).

## Contributing

Want to contribute to `rlberry`? Please check [our contribution guidelines](https://rlberry-py.github.io/rlberry/stable/contributing.html). **If you want to add any new agents or environments, do not hesitate
to [open an issue](https://github.com/rlberry-py/rlberry/issues/new/choose)!**
