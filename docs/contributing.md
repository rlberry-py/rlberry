(contributing)=

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

## Guidelines for docstring

* Follow the [numpydoc docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html).

## Have a video for an example in the documentation

To generate the videos for the examples, cd to the docs folder  and then use `make video`.

Here is a template of the python script of a video example:
```python
"""
===============
Some good title
===============
Some explanation text of what you are doing

.. video:: ../video_plot_my_experiment.mp4
   :width: 600
"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_my_experiment.jpg'

# Write here the code that generates the video


# Save the video
video = env.save_video("../docs/_video/video_plot_my_experiment.mp4", framerate=10)
```

For a video to be automatically compiled with `make video`, you must follow this
template replacing the "my_experiment" with the name of your example. It may be
useful to change the framerate in the last line of the code to have a faster or
slower framerate depending on your environment.

After running `make video`, you should have your video available in `docs/_video`
you should add this video to the git repo with `git add docs/_video/video_plot_my_experiment.mp4`
and `git add docs/thumbnails/video_plot_my_experiment.jpg` to add the associated thumbnail.

Then just push the new examples, the mp4 and the jpg files, they should be included in the doc.


## Guidelines for new agents

* Create a folder for the agent `rlberry/agents/agent_name`.
* Create `rlberry/agents/agent_name/__init__.py`.
* Write a test to check that the agent is running `rlberry/agents/test_agent_name.py`.
* Write an example `examples/demo_agent_name.py`.

### Agent code template

The template below gives the general structure that the Agent code must follow. See more options in the abstract `Agent`
class (`rlberry/agents/agent.py`).

```python

class MyAgent(Agent):
    name = "MyAgent"

    def __init__(self,
                 env,
                 param_1,
                 param_2,
                 param_n,
                 **kwargs):
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
        param_1 = trial.suggest_categorical('param_1',
                                            [1, 4, 8, 16, 32, 64])
        # ... and param_2 could be a learning_rate
        param_2 = trial.suggest_loguniform('param_2', 1e-5, 1)
        return {
            'param_1': param_1,
            'param_2': param_2,
        }
```

### Implementation notes

* When inheriting from the `Agent` class, make sure to call `Agent.__init__(self, env, **kwargs)` using `**kwargs` in
  case new features are added to the base class.

Infos, errors and warnings are printed using the `logging` library.

* From `gym` to `rlberry`:
    * `reseed` (rlberry) should be called instead of `seed` (gym). `seed` keeps compatilibity with gym, whereas `reseed`
      uses the unified seeding mechanism of `rlberry`.

## Guidelines for logging

* `logger.info()`, `logger.warning()` and `logger.error()` should be used inside the `rlberry` package, rather
  than `print()`.
* The desired level of verbosity can be chosen by calling `configure_logging`, e.g.:

```python
from rlberry.utils.logging import configure_logging

configure_logging(level="DEBUG")
configure_logging(level="INFO")
# etc
```

* `print()` statements can be used outside `rlberry`, e.g., in scripts and notebooks.
