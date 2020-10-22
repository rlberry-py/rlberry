# rlberry - A Reinforcement Learning Library for Research and Education 

![pytest](https://github.com/rlberry-py/rlberry/workflows/test/badge.svg)

# Main differences with other libraries

Our goals:
* Structured documentation/tutorial/examples for each algorithm (inspired by sklearn). Good for RL courses.
* Modular code: the implementation of each algorithm must be modular enough to allow improvements/modifications (useful for research).
* Implement traditional RL algorithms so that we can compare them to deep algorithms. Before solving large scale problems with deep RL, we can validate the algorithms in small scale environments, where traditional RL works very well. Faster prototyping for deep algorithms. 
* Module to automatically compare/benchmark algorithms. 
* Avoid seed hacking with a unified seeding mechanism! 


# Install

Creating a virtual environment and installing:

```
conda create -n rlberry python=3.7
conda activate rlberry
pip install -e .
```

# Tests

To run tests, run `pytest`.

With coverage: install and run pytest-cov
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

* Convention for verbose:
    * `verbose<0`: nothing is printed
    * `verbose=0`: print only erros and importatnt warnings
    * `verbose>1`: print progress messages
