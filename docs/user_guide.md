(user_guide)=


# User Guide

You can find a **compacted version** [here](#compacted-version).

## Introduction
Welcome to rlberry.
Use rlberry's [ExperimentManager](experimentManager_page) to train, evaluate and compare rl agents.
Like other popular rl libraries, rlberry also provides basic tools for plotting, multiprocessing and logging  <!-- TODO :(add refs)-->. In this user guide, we take you through the core features of rlberry and illustrate them with [examples](/auto_examples/index) and [API documentation](/api) .

To run all the examples, you will need to install "[rlberry-research](https://github.com/rlberry-py/rlberry-research)" and "[rlberry-scool](https://github.com/rlberry-py/rlberry-scool)" too.
 <!-- TODO : Add some code with the best solution to install them: poetry?, pip?, github link ??? -->

 [rlberry-research](https://github.com/rlberry-py/rlberry-research) :
 It's the repository where our research team keeps some agents, environments, or tools compatible with rlberry. It's a permanent "work in progress" repository, and some code may be not maintained anymore.

[rlberry-scool](https://github.com/rlberry-py/rlberry-scool) :
It's the repository used for teaching purposes. These are mainly agents or very basic environments, in a version that makes it easier for students to learn.


## Set up an experiment
Some text about an experiment: a comparison of 2 rl agents on a given environment.  <!-- TOCHECK : it is the same as quickstart.md -->
### Environment
This is the world with which the agent interacts. The agent can observe this environment, and can perform actions to modify it (but cannot change its rules). [(Wikipedia)](https://en.wikipedia.org/wiki/Reinforcement_learning) The environment is typically stated in the form of a Markov decision process (MDP).
You can find the guide for Environment [here](environment_page).

Some text about MDPs.(?<!-- TOCHECK :plus tard -->)
### Agent <!-- TOCHECK :plus tard -->
In Reinforcement learning, the Agent is the entity to train to solve an environment. It's able interact with the environment: observe, take actions, and learn through trial and error.
You can find the guide for Agent [here](agent_page).

### ExperimentManager
This is one of the core element in rlberry. The ExperimentManager allow you to easily make an experiment between an Agent and an Environment. It's use to train, optimize hyperparameters, evaluate and gather statistics about an agent.
You can find the guide for ExperimentManager [here](experimentManager_page).
### Logging
Logging is used to keep a trace of the experiments. It's include runing informations, data, and results.
You can find the guide for Logging [here](logging_page).
### Results analysis
In construction
## Experimenting with Deep agents
### Torch Agents   <!-- TOCHECK :plus tard -->
In construction
### Policy and Value Networks<!-- TOCHECK :plus tard -->
In construction
<!--already in Agent_page ### Stable Baselines 3 -->
## Experimenting with Bandits. <!-- TOCHECK :plus tard -->
In construction
## Reproducibility
### Seeding
In rlberry you can use a seed to generate pseudo-"random number". Most of the time, it allow you to re-run the same algorithm with the same pseudo-"random number", and make your experiment reproducible.You can find the guide for Seeding [here](seeding_page).

### Saving and Loading Experiment
You can save and load your experiments.
It could be useful in many way :
- don't repeat the training part every time.
- continue a previous training (or doing checkpoint).

You can find the guide for Saving and Loading [here](save_load_page).
### Saving and Loading Agents <!-- TOCHECK :plus tard -->
In construction
### Saving and Loading Data <!-- TOCHECK :plus tard -->
In construction

<!--
---------------------------------------

regarder/expliquer les LoadResults.py  (dans rlberry/experiment et ..../test

----------------------------------------)
 -->

## Advanced Usage<!-- TOCHECK :plus tard -->
### Custom Agents <!-- TOCHECK :plus tard -->
In construction
### Custom Environments<!-- TOCHECK :plus tard -->
In construction
### Transfer Learning<!-- TOCHECK :plus tard -->
In construction


# Compacted version
## Set up an experiment
- [Environment](environment_page)
- [Agent](agent_page)
- [ExperimentManager](experimentManager_page)
- [Logging](logging_page).
- [Results analysis (In construction)]()
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
- [Transfer Learning (In construction)]()
