(user_guide)=


# User Guide

## Introduction
Welcome to rlberry.
Use rlberry's [ExperimentManager](experimentManager_page) to train, evaluate and compare rl agents.
Like other popular rl libraries, rlberry also provides basic tools for plotting, multiprocessing and logging  <!-- TODO :(add refs)-->. In this user guide, we take you through the core features of rlberry and illustrate them with [examples](/auto_examples/index) and [API documentation](/api) .

To run all the examples, you will need to install other libraries like "[rlberry-scool](https://github.com/rlberry-py/rlberry-scool)" (and others).
 <!-- TODO : Add some code with the best solution to install them: poetry?, pip?, github link ??? -->
The easiest way to do it is :
```none
pip install rlberry[torch,extras]
pip install rlberry-scool
```

[rlberry-scool](https://github.com/rlberry-py/rlberry-scool) :
It's the repository used for teaching purposes. These are mainly basic agents and environments, in a version that makes it easier for students to learn.

You can find more details about installation [here](installation)!

 You can find our quick starts here :
 - [RL quickstart](quick_start)
 - [Deep RL quickstart](TutorialDeepRL).

## Set up an experiment
```{include} templates/nice_toc.md
```

```{toctree}
:maxdepth: 2
basics/userguide/environment.md
basics/userguide/agent.md
basics/userguide/experimentManager.md
basics/userguide/logging.md
```
- Results analysis & visualization (In construction)
## Experimenting with Deep agents
[(In construction)](https://github.com/rlberry-py/rlberry/issues/459)
## Reproducibility
```{toctree}
:maxdepth: 2
basics/userguide/seeding.md
basics/userguide/save_load.md
```
- Save & Load Agents (In construction)
- Save & Load Data (In construction)
## Advanced Usage
```{toctree}
:maxdepth: 2
basics/userguide/adastop.md
basics/comparison.md
basics/userguide/external_lib.md
```
- Custom Agents (In construction)
- Custom Environments (In construction)
- Transfer Learning (In construction)

# Contributing to rlberry
If you want to contribute to rlberry, check out [the contribution guidelines](contributing).
