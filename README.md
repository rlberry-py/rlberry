# rlberry 

![pytest](https://github.com/rlberry-py/rlberry/workflows/test/badge.svg)


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