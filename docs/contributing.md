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
- ...

Please read the rest of this page for more information on how to contribute.


# Submitting a bug report or a feature request

We use GitHub issues to track all bugs and feature requests; feel free to open an issue if you have found a bug or wish to see a feature implemented.

In case you experience issues using this package, do not hesitate to submit a ticket to the Bug Tracker. You are also welcome to post feature requests or pull requests.

It is recommended to check that your issue complies with the following rules before submitting:

* Verify that your issue is not being currently addressed by other issues or pull requests.
* If you are submitting an algorithm or feature request, please verify that the algorithm fulfills our new algorithm requirements.
* If you are submitting a bug report, we strongly encourage you to follow the guidelines in How to make a good bug report.

## How to make a good bug report

When you submit an issue to Github, please do your best to follow these guidelines! This will make it a lot easier to provide you with good feedback:

* The ideal bug report contains a short reproducible code snippet, this way anyone can try to reproduce the bug easily (see this for more details). If your snippet is longer than around 50 lines, please link to a gist or a github repo.
* If not feasible to include a reproducible snippet, please be specific about what agents, parameters and environments are involved.
* If an exception is raised, please provide the full traceback.
* Please ensure all code snippets and error messages are formatted in appropriate code blocks. See [Creating and highlighting code blocks](https://docs.github.com/en/github/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks) for more details.

# Contributing code

```{admonition} Note

To avoid duplicating work, it is highly advised that you search through the issue tracker and the PR list. If in doubt about duplicated work, or if you want to work on a non-trivial feature, it’s recommended to first open an issue in the issue tracker to get some feedback from core developers.

One easy way to find an issue to work on is by applying the “help wanted” label in your search. This lists all the issues that have been unclaimed so far.

```



## How to contribute -- git crash-course

The preferred way to contribute to rlberry is to fork the main repository on GitHub, then submit a “pull request” (PR).

In the first few steps, we explain how to locally install rlberry, and how to set up your git repository:

1. [Create an account](https://github.com/join) on GitHub if you do not already have one.
2. Fork the [project repository](https://github.com/rlberry-py/rlberry): click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account. For more details on how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).
3. Clone your fork of rlberry repo from your GitHub account to your local disk:
    ```bash
    git clone https://github.com/YOUR_LOGIN/rlberry  # add --depth 1 if your connection is slow
    cd rlberry
    ```
4. Install the package locally with pip
    ```bash
    pip install -e . --user
    ```
5. Install the development dependencies
    ```bash
    pip install pytest pytest-cov flake8 black
    ```
6. Add the upstream remote. This saves a reference to the main rlberry repository, which you can use to keep your repository synchronized with the latest changes:
    ```bash
    git remote add upstream https://github.com/rlberry-py/rlberry
    ```
    You should now have a working installation of rlberry, and your git repository properly configured. The next steps now describe the process of modifying code and submitting a PR:
7. Synchronize your main branch with the upstream/main branch, more details on [GitHub Docs](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork):
    ```bash
    git checkout main
    git fetch upstream
    git merge upstream/main
    ```
8. Create a feature branch to hold your development changes:
    ```bash
    git checkout -b my_feature
    ```
    and start making changes. Always use a feature branch. It’s good practice to never work on the main branch!
9. Develop the feature on your feature branch on your computer, using Git to do the version control. When you’re done editing, add changed files using git add and then git commit:
    ```bash
    git add modified_files
    git commit
    ```
    to record your changes in Git, then push the changes to your GitHub account with:
    ```bash
    git push -u origin my_feature
    ```
10. Follow [these](https://help.github.com/articles/creating-a-pull-request-from-a-fork) instructions to create a pull request from your fork. This will send an email to the committers.


````{admonition} Note
It is often helpful to keep your local feature branch synchronized with the latest changes of the main rlberry repository:
```bash
git fetch upstream
git merge upstream/main
```
Subsequently, you might need to solve the conflicts. You can refer to the [Git documentation related to resolving merge conflict using the command line](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/). You may also want to use
```bash
git rebase upstream/main
```
to have the same commit has the main repo.
````

## Pull request checklist

Before a PR can be merged, it needs to be approved. Please prefix the title of your pull request with [MRG] if the contribution is complete and should be subjected to a detailed review. An incomplete contribution – where you expect to do more work before receiving a full review – should be prefixed [WIP] (to indicate a work in progress) and changed to [MRG] when it matures. WIPs may be useful to: indicate you are working on something to avoid duplicated work, request broad review of functionality or API, or seek collaborators. WIPs often benefit from the inclusion of a [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments) in the PR description.

1. Give your PR a helpful title
2. Make sure your PR passes the test. You can check this by typing `pytest` from the main folder `rlberry`, or you can run a particular test by running for instance `pytest rlberry/agents/tests/` for instance, replacing `rlberry/agents/tests/` by the folder of the test you want to run.
3. Make sure your code is properly commented and documented, and make sure the documentation renders properly. To build the documentation, please refer to our [Documentation guidelines](documentation). The CI will also build the docs: please refer to Generated documentation at PR time.
4. Tests are necessary for enhancements to be accepted. You must include tests to verify the correct behavior of the fix or feature.
5. Run black to auto-format your code.
    ```bash
    black .
    ```
6. Make sure that your PR does not add PEP8 violations. To check the code that you changed, you can use
    ```bash
    git diff upstream/main -u -- "*.py" | flake8 --diff
    ```
## Continuous integration (CI)


* Azure pipelines are used for testing rlberry on Linux, Mac and Windows, with different dependencies and settings.
* Readthedocs is used to build the docs for viewing.

Please note that if you want to skip the CI (azure pipeline is long to run), use `[ci skip]` in the description of the commit.

(contributing)=

# Documentation

We are glad to accept any sort of documentation: function docstrings, reStructuredText or markdown documents (like this one), tutorials, examples, etc. reStructuredText and markdown documents live in the source code repository under the docs/ directory.

You can edit the documentation using any text editor, and then generate the HTML output by typing make from the docs/ directory. Alternatively, make html may be used to generate the documentation with the example gallery (which takes quite some time). The resulting HTML files will be placed in _build/html/stable and are viewable in a web browser.

## Building the documentation


Building the documentation requires installing some additional packages:
```bash
pip install sphinx sphinx-gallery numpydoc myst-parser --user
```
To build the documentation, you need to be in the doc folder:
```bash
cd docs
```
You may only need to generate the full website, without the example gallery:
```bash
make
```
The documentation will be generated in the _build/html/stable directory. To also generate the example gallery you can use:
```bash
make html
```
This will run all the examples, which takes a while. If you only want to generate a few examples, you can use:
```bash
EXAMPLES_PATTERN=your_regex_goes_here make html
```
## Guidelines for writing documentation
### Guidelines for docstring

Please follow the [numpydoc docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html), the minimal requirements should be to include the short description, parameters and  return section of the docstring.

### Examples

The examples gallery is constructed using sphinx-gallery, see its [documentation](https://sphinx-gallery.readthedocs.io/en/latest/) for more information.

### Other documentation

The documentation is done using sphinx, each article can be written either in reStructuredText (rst) format or in markdown. For markdown support, we use myst-parser (see its [documentation](https://myst-parser.readthedocs.io/en/latest/using/intro.html)).

If you need to cross-reference your documentations, you can use
for rst:
```
.. _nameref:

some text.
```
and for markdown:
```
(contributing)=

some text.
```
If you want to look at some examples, you can look at doc/index.rst file for rst file example and the present file (contributing.md) for example of markdown syntax.

# Guidelines for new agents
=======

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

.. In the path for the video described before, use an additional ".." if your
    experiment is in a sub-folder of the examples folder.
"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_my_experiment.jpg'

# Write here the code that generates the video


# Save the video
video = env.save_video("../docs/_video/video_plot_my_experiment.mp4", framerate=10)
```

For a video to be automatically compiled with `make video`, you must follow this
template replacing the "my_experiment" with the name of your example. It may be
useful to change the framerate in the last line of the code to have a faster or
slower frame rate depending on your environment.

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
    * `reseed` (rlberry) should be called instead of `seed` (gym). `seed` keeps compatibility with gym, whereas `reseed`
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


# Acknowledgements

Part of this page was copied from [scikit-learn contributing guideling](https://scikit-learn.org/dev/developers/contributing.html#documentation).
=======
