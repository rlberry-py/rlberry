(user_guide)=


# User Guide

## Introduction
Welcome to rlberry.
Use rlberry's [ExperimentManager](experimentManager_page) to train, evaluate and compare rl agents.
Like other popular rl libraries, rlberry also provides basic tools for plotting, multiprocessing and logging  <!-- TODO :(add refs)-->. In this user guide, we take you through the core features of rlberry and illustrate them with [examples](/auto_examples/index) and [API documentation](/api) .

To run all the examples, you will need to install other libraries like "[rlberry-research](https://github.com/rlberry-py/rlberry-research)" and "[rlberry-scool](https://github.com/rlberry-py/rlberry-scool)" (and others).
 <!-- TODO : Add some code with the best solution to install them: poetry?, pip?, github link ??? -->
The easiest way to do it is :
```none
pip install rlberry[torch,extras]
pip install git+https://github.com/rlberry-py/rlberry-research.git
```

 [rlberry-research](https://github.com/rlberry-py/rlberry-research) :
 It's the repository where our research team keeps some agents, environments, or tools compatible with rlberry. It's a permanent "work in progress" repository, and some code may be not maintained anymore.

[rlberry-scool](https://github.com/rlberry-py/rlberry-scool) :
It's the repository used for teaching purposes. These are mainly basic agents and environments, in a version that makes it easier for students to learn.

You can find more details about installation [here](installation)

## Set up an experiment
- [Environment](environment_page)
- [Agent](agent_page)
- [ExperimentManager](experimentManager_page)
- [Logging](logging_page).
- [Results analysis & visualization (In construction)]()
## Experimenting with Deep agents
- [Torch Agents (In construction)]()
- [Policy and Value Networks (In construction)]()
- [Experimenting with Bandits (In construction)]()
## Reproducibility
- [Seeding](seeding_page)
- [Save & Load Experiment](save_load_page)
- [Save & Load Agents (In construction)]()
- [Save & Load Data (In construction)]()
## Advanced Usage
- [Custom Agents (In construction)]()
- [Custom Environments (In construction)]()
- [Using extrenal libraries](external) (like [Stable Baselines](stable_baselines) and [Gymnasium](Gymnasium_ancor))
- [Transfer Learning (In construction)]()
- [AdaStop(In construction)]()
