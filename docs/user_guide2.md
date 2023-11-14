(user_guide2)=


# User Guide
## Introduction
Welcome to rlberry. Use rlberry's ExperimentManager (add ref) to train, evaluate and compare rl agents. In addition to
the core ExperimentManager (add ref), rlberry provides the user with a set of bandit (add ref), tabular rl (add ref), and
deep rl agents (add ref) as well as a wrapper for stablebaselines3 (add link, and ref) agents.
Like other popular rl libraries, rlberry also provides basic tools for plotting, multiprocessing and logging (add refs).
In this user guide, we take you through the core features of rlberry and illustrate them with examples (add ref) and API documentation (add ref).
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
### Stable Baselines 3
## Experimenting with Bandits. <!-- TOCHECK :plus tard -->
## Reproducibility
### Seeding
### Saving and Loading Agents
### Saving and Loading Data
## Advanced Usage<!-- TOCHECK :plus tard -->
### Custom Agents <!-- TOCHECK :plus tard -->
### Custom Environments<!-- TOCHECK :plus tard -->
### Transfer Learning<!-- TOCHECK :plus tard -->
