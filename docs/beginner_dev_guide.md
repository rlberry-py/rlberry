(dev_guide)=

## How to contribute
### git crash-course

The preferred way to contribute to rlberry is to fork the main repository on GitHub, then submit a “pull request” (PR).

In the first few steps, we explain how to locally install rlberry, and how to set up your git repository:

1. [Create an account](https://github.com/join) on GitHub if you do not already have one.
2. Fork the [project repository](https://github.com/rlberry-py/rlberry): click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account. For more details on how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).
3. Clone your fork of rlberry repo from your GitHub account to your local disk:
    ```bash
    git clone https://github.com/YOUR_LOGIN/rlberry  # add --depth 1 if your connection is slow
    cd rlberry
    ```
4. Install the full dependencies
    ```bash
    pip install -r requirement.txt
    ```
5. Install the package locally with pip
    ```bash
    pip install -e . --user
    ```
6. Install the development dependencies
    ```bash
    pip install pytest pytest-cov flake8 black
    ```
7. Add the upstream remote. This saves a reference to the main rlberry repository, which you can use to keep your repository synchronized with the latest changes:
    ```bash
    git remote add upstream https://github.com/rlberry-py/rlberry
    ```
    You should now have a working installation of rlberry, and your git repository properly configured. The next steps now describe the process of modifying code and submitting a PR:
8. Synchronize your main branch with the upstream/main branch, more details on [GitHub Docs](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork):
    ```bash
    git checkout main
    git fetch upstream
    git merge upstream/main
    ```
9. Create a feature branch to hold your development changes:
    ```bash
    git checkout -b my_feature
    ```
    and start making changes. Always use a feature branch. It’s good practice to never work on the main branch!
10. Develop the feature on your feature branch on your computer, using Git to do the version control. When you’re done editing, add changed files using git add and then git commit:
    ```bash
    git add modified_files
    git commit
    ```
    to record your changes in Git, then push the changes to your GitHub account with:
    ```bash
    git push -u origin my_feature
    ```
11. Follow [these](https://help.github.com/articles/creating-a-pull-request-from-a-fork) instructions to create a pull request from your fork. This will send an email to the committers.


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

### Pre-commit (optional)

You may want to use [pre-commit](https://pre-commit.com/) to check some issues
with your code. pre-commit is a software that will automatically run every time you
do `git commit` and it will do some style changes (black, flake8...) before the commit.

To use, do

```
pip install pre-commit
```

Then, in the rlberry folder,

```
pre-commit install
```

and then you are done. When next you do a commit, some checks will be run on the
changes you made and if your code need some reformatting, it will automatically
be done for you, you will only need to recommit to add the changes made by pre-commit.
Once pre-commit is installed, if you want to skip it you can with
`git commit . -m 'quick fix' --no-verify`.

It can also be useful to use `autopep8 -a -a -i yourfile` to fix flake8 issues.


### Pull request checklist

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
### Continuous integration (CI)


* Azure pipelines are used for testing rlberry on Linux, Mac and Windows, with different dependencies and settings.
* Readthedocs is used to build the docs for viewing.

Please note that if you want to skip the CI (azure pipeline is long to run), use `[ci skip]` in the description of the commit.

### Building examples

The examples gallery is constructed using sphinx-gallery, see its [documentation](https://sphinx-gallery.readthedocs.io/en/latest/) for more information.

### Markdown and link between documentation pages.

The documentation is done using sphinx, each article can be written either in reStructuredText (rst) format or in markdown. For markdown support, we use myst-parser (see its [documentation](https://myst-parser.readthedocs.io/en/latest/using/intro.html)). The examples and docstrings on the other hand use only rst.

If you need to cross-reference your documentations, you can use
for rst:
```
.. _nameref:

some text.
```
and for markdown:
```
(namered)=

some text.
```
If you want to look at some examples, you can look at docs/index.rst file for rst file example and the present file (contributing.md) for example of markdown syntax.

### Have a video for an example in the documentation

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

## Acknowledgements

Part of this page was copied from [scikit-learn contributing guideling](https://scikit-learn.org/dev/developers/contributing.html#documentation).
