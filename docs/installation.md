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

minimal version :
```bash
$ pip install rlberry
```


Recommanded version (more adapted for RL usage):
```bash
$ pip install rlberry[extras]
```
** `extras` allow to install : **
`optuna` : Optuna is an automatic hyperparameter optimization software framework. More information [here](https://optuna.org/)
`ffmpeg-python` : Python bindings for FFmpeg - with complex filtering support. A complete, cross-platform solution to record, convert and stream audio and video. More information [here](https://pypi.org/project/ffmpeg-python/)
`scikit-fda` : This package offers classes, methods and functions to give support to Functional Data Analysis in Python. More information [here](https://fda.readthedocs.io/en/latest/)

DeepRL version :
```bash
$ pip install rlberry[torch,extras]
```
** `torch` allow to install : **
`ale-py` : The Arcade Learning Environment (ALE) is a simple framework that allows researchers and hobbyists to develop AI agents for Atari 2600 games. More information [here](https://pypi.org/project/ale-py/)
`opencv-python` : Wrapper package for OpenCV python bindings. OpenCV provides a real-time optimized Computer Vision library. More information [here](https://pypi.org/project/opencv-python/)
`stable-baselines3` : Stable Baselines3 (SB3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch. More information [here](https://stable-baselines3.readthedocs.io/en/master/)
`tensorboard` : TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. More information [here](https://www.tensorflow.org/tensorboard/get_started)
`torch` : The torch package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors. More information [here](https://pytorch.org/docs/stable/torch.html)



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
