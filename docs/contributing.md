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


## Documentation

We are glad to accept any sort of documentation: function docstrings, reStructuredText or markdown documents (like this one), tutorials, examples, etc. reStructuredText and markdown documents live in the source code repository under the docs/ directory.

### Building the documentation
In the following section, we assume that you are in the main rlberry directory.

Building the documentation requires installing some additional packages:
```bash
pip install -r docs/requirements.txt --user
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

## Guidelines for new agents

* Create a folder for the agent `rlberry/agents/agent_name`.
* Create `rlberry/agents/agent_name/__init__.py`.
* Write a test to check that the agent is running `rlberry/agents/test_agent_name.py`.
* Write an example `examples/demo_agent_name.py`.

## Agent code template

The template below gives the general structure that the Agent code must follow. See more options in the abstract `Agent`
class (`rlberry/agents/agent.py`).

```python
class MyAgent(Agent):
    name = "MyAgent"

    def __init__(self, env, param_1, param_2, param_n, **kwargs):
        Agent.__init__(self, env, **kwargs)

    def fit(self, budget: int):
        """
        ** Must be implemented. **

        Trains the agent, given a computational budget (e.g. number of steps or episodes).
        """
        # code to train the agent
        # ...
        pass

    def eval(self, **kwargs):
        """
        ** Must be implemented. **

        Evaluates the agent (e.g. Monte-Carlo policy evaluation).
        """
        return 0.0

    @classmethod
    def sample_parameters(cls, trial):
        """
        ** Optional **

        Sample hyperparameters for hyperparam optimization using
        Optuna (https://optuna.org/).

        Parameters
        ----------
        trial: optuna.trial
        """
        # Note: param_1 and param_2 are in the constructor.

        # for example, param_1 could be the batch_size...
        param_1 = trial.suggest_categorical("param_1", [1, 4, 8, 16, 32, 64])
        # ... and param_2 could be a learning_rate
        param_2 = trial.suggest_loguniform("param_2", 1e-5, 1)
        return {
            "param_1": param_1,
            "param_2": param_2,
        }
```

### Implementation notes

* When inheriting from the `Agent` class, make sure to call `Agent.__init__(self, env, **kwargs)` using `**kwargs` in
  case new features are added to the base class.

Infos, errors and warnings are printed using the `logging` library.

* From `gym` to `rlberry`:
    * `reseed` (rlberry) should be called instead of `seed` (gym). `seed` keeps compatibility with gym, whereas `reseed`
      uses the unified seeding mechanism of `rlberry`.

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
