<img src="logo/logo_wide.svg" width="50%">

# A Reinforcement Learning Library for Research and Education 

![pytest](https://github.com/rlberry-py/rlberry/workflows/test/badge.svg)

# Philosophy

* Detailed documentation and comprehensible tutorial/examples (Jupyter Notebook) for each implemented algorithm.
* Provide a very general interface for agents, that
    * puts **minimal constraints** on the agent code (=> making it easy to include new algorithms and modify existing ones);
    * allows comparison between agents using a simple and unified evaluation interface (=> making it easy, for instance, to compare deep and "traditional" RL algorithms).
* Unified seeding mechanism: define only one global seed, from which all other seeds will inherit, enforcing independence of the random number generators (=> avoid seed "optimization"!).
* Simple interface for creating and **rendering** new environments. 


# Install

To install, first create a virtual environment using Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (suggested):

```
conda create -n rlberry python=3.7
conda activate rlberry
pip install -e .
```

Or you can also install directly (not suggested):

```
python3 -m pip install -e .
```

# Tests

To run tests, run `pytest`. To run tests with coverage, install and run `pytest-cov`:

```
pip install pytest-cov
bash run_tests.sh
```

See coverage report in `cov_html/index.html`.


# Notes

* To save videos, installing [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) is required:

```
pip install ffmpeg-python
```

* Convention for verbose in the agents:
    * `verbose=0`: nothing is printed
    * `verbose>1`: print progress messages

Errors and warnings are printed using the `logging` library.


# Some design principles

* Agents should not keep references to objects outside of their scope (e.g. reference to an environment supposed to be used/modified elsewhere). For instance, the abstract Agent class
makes a deep copy of the input environment.
