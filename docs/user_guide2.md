(user_guide2)=


# User Guide
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
### Analyse the results
## Experimenting with Deep agents
### Torch Agents   <!-- TOCHECK :plus tard -->
### Policy and Value Networks<!-- TOCHECK :plus tard -->
<!--already in Agent_page ### Stable Baselines 3 -->
## Experimenting with Bandits. <!-- TOCHECK :plus tard -->
## Reproducibility
### Seeding
### Saving and Loading Agents
### Saving and Loading Data
## Advanced Usage<!-- TOCHECK :plus tard -->
### Custom Agents <!-- TOCHECK :plus tard -->
### Custom Environments<!-- TOCHECK :plus tard -->
### Transfer Learning<!-- TOCHECK :plus tard -->
