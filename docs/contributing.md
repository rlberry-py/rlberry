# Contributing

Currently, we are accepting the following forms of contributions:

- Bug reports (open
  an [Issue](https://github.com/rlberry-py/rlberry/issues/new?assignees=&labels=&template=bug_report.md&title=)
  indicating your system information, the current behavior, the expected behavior, a standalone code to reproduce the
  issue and other information as needed).
- Pull requests for bug fixes.
- Improvements/benchmarks for deep RL agents.
- Documentation improvements.
- New environments.
- New agents.
- ...

Please read the rest of this page for more information on how to contribute to
rlberry and look at our [beginner developer guide](dev_guide) if you have questions
about how to use git, do PR or for more informations about the documentation.

For a PR to be accepted it must pass all the tests when the label 'ready for CI' is set in the PR. The label 'ready for CI' trigger additional tests to be done on the PR and should be set when the PR is ready.

## Documentation

We are glad to accept any sort of documentation: function docstrings, reStructuredText or markdown documents (like this one), tutorials, examples, etc. reStructuredText and markdown documents live in the source code repository under the docs/ directory.

### Building the documentation
In the following section, we assume that you are in the main rlberry directory.

Building the documentation requires installing some additional packages:
```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install --with dev,doc,torch,extras --sync
```
To build the documentation, you need to be in the docs folder:
```bash
cd docs
```
You may only need to generate the full website, without the example gallery:
```bash
make
```
The documentation will be generated in the `_build/html` directory. To also generate the example gallery you can use:
```bash
make html
```
This will run all the examples, which takes a while. If you only want to generate a few examples, you can use:
```bash
EXAMPLES_PATTERN=your_regex_goes_here make html
```

### Tests

We use `pytest` for testing purpose. We cover two types of test: the coverage tests that check that every algorithm does what is intended using as little computer resources as possible and what we call long tests. The long tests are here to test the performance of our algorithms. These tests are too long to be run on the azure pipeline and hence they are not run automatically at each PR. Instead they can be launched locally to check that the main algorithms of rlberry perform as efficiently as previous implementation.

Running tests from root of repository:
```bash
pytest .
```

Running long tests:
```bash
pytest -s long_tests/**/ltest*.py
```

Tests files must be named `test_[something]` and belong to one of the `tests` directory. Long test files must be names  `ltest_[something]` and belong to the `long_tests` directory.

### Guidelines for docstring


Please follow the [numpydoc docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html), the minimal requirements should be to include the short description, parameters and  return section of the docstring.


## Guidelines for logging

* `logger.info()`, `logger.warning()` and `logger.error()` should be used inside the `rlberry` package, rather
  than `print()`.
* The desired level of verbosity can be chosen by calling `set_level`, e.g.:

```python
from rlberry.utils.logging import set_level

set_level(level="DEBUG")
set_level(level="INFO")
# etc
```

* `print()` statements can be used outside `rlberry`, e.g., in scripts and notebooks.
