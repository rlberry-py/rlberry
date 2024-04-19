(installation)=

# Installation
First, we suggest you to create a virtual environment using
[Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
$ conda create -n rlberry
$ conda activate rlberry
```
## Latest version (0.7.2)
Install the latest version for a stable release.

```bash
$ pip install rlberry
```

## Options
To install rlberry with more options, you can use ``pip install rlberry[xxxxxxxx]``, with `xxxxxxxx` as :

- `torch` to install `opencv-python, ale-py, stable-baselines3, tensorboard, torch`
- `extras` to install `optuna, ffmpeg-python, scikit-fda`

(for dev)

- `doc` to install `sphinx, sphinx-gallery, sphinx-math-dollar, numpydoc, myst-parser, sphinxcontrib-video, matplotlib`

## Development version
Install the development version to test new features.

```bash
$ pip install rlberry@git+https://github.com/rlberry-py/rlberry.git
```

<span>&#9888;</span> **warning :** <span>&#9888;</span>

For `zsh` users, `zsh` uses brackets for globbing, therefore it is necessary to add quotes around the argument,
e.g. ```pip install 'rlberry@git+https://github.com/rlberry-py/rlberry.git'```.

## Previous versions
If you used a previous version in your work, you can install it by running

```bash
$ pip install rlberry@git+https://github.com/rlberry-py/rlberry.git@{TAG_NAME}
```

replacing `{TAG_NAME}` by the tag of the corresponding version,
e.g., ```pip install rlberry@git+https://github.com/rlberry-py/rlberry.git@v0.1```
to install version 0.1.

<span>&#9888;</span> **warning :** <span>&#9888;</span>

For `zsh` users, `zsh` uses brackets for globbing, therefore it is necessary to add quotes around the argument,
e.g. ```pip install 'rlberry@git+https://github.com/rlberry-py/rlberry.git@v0.1'```.

## Deep RL agents
Deep RL agents require extra libraries, like PyTorch.

* PyTorch agents:

```bash
$ pip install rlberry[torch]@git+https://github.com/rlberry-py/rlberry.git
```

<span>&#9888;</span> **warning :** <span>&#9888;</span>

For `zsh` users, `zsh` uses brackets for globbing, therefore it is necessary to add quotes around the argument,
e.g. ```pip install 'rlberry[torch_agents]@git+https://github.com/rlberry-py/rlberry.git'```.
