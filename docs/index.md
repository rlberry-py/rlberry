(index)=

```{image} ../assets/logo_wide.svg
:align: center
:width: 50%
:alt: rlberry logo
```

## An RL Library for Research and Education
**Writing reinforcement learning algorithms is fun!** *But after the fun, we have
lots of boring things to implement*: run our agents in parallel, average and plot results,
optimize hyperparameters, compare to baselines, create tricky environments etc etc!

[rlberry](https://github.com/rlberry-py/rlberry) **is here to make your life easier** by doing all these things with a few lines of code, so that you can spend most of your time developing agents.

We provide you a number of tools to help you achieve **reproducibility**, **statistically comparisons** of RL agents, and **nice visualization**.

 If you begin with [rlberry](https://github.com/rlberry-py/rlberry), **check our** [RL quickstart](quick_start) **and our** [Deep RL quickstart](TutorialDeepRL).


``````{grid} 3
:gutter: 3 3 3 3
```{grid-item-card} Comprehensive workflow for RL research
:link: quick_start
:link-type: ref
:img-bottom: thumbnails/experiment_manager_diagram.png
:width: 90

Tools designed to facilitate RL experimentation.
```
```{grid-item-card} Statistical tests
:link: comparison_page
:link-type: ref
:img-bottom: thumbnails/adastop_boxplots.png
:width: 90

Statistical tools for agent comparison.

```
```{grid-item-card} Visualization
:link: visu_userguide
:link-type: ref
:img-bottom: thumbnails/output_9_3.png
:width: 90

Proper smoothing and plot of confidence intervals.
```
```{grid-item-card} Teaching tools
:link: https://github.com/rlberry-py/rlberry-scool
:img-bottom: thumbnails/chain_thumb.jpg
:width: 90

Simple environments and tools for use in teaching RL.
```
```{grid-item-card} Hackable library
:link: https://rlberry-py.github.io/rlberry/basics/userguide/agent.html#create-your-own-agent
:width: 90
:img-bottom: thumbnails/code.png


Code is easy to read and easy to change.
```
```{grid-item-card} Compatible
:link: user_guide
:link-type: ref
:width: 90

Uses *stable-baselines*, *tensorboard*, *gymnasium*, and you can easily write your wrapper to use your own RL library. Visualization and statistical tool can be used with dataframes as input/output in order to be library-independent.
```
``````

## Documentation Contents
You can find main documentation here :
- [Installation](installation)
- [User Guide](user_guide)
- [Examples](examples)
- [API](api)
- [Changelog](changelog)


## Contributing to rlberry
If you want to contribute to rlberry, check out [the contribution guidelines](contributing).

## rlberry main features

### ExperimentManager
This is one of the core element in [rlberry](https://github.com/rlberry-py/rlberry). The [ExperimentManager](rlberry.manager.experiment_manager.ExperimentManager) allow you to easily make an experiment between an [Agent](agent_page) and an [Environment](environment_page). It's use to train, optimize hyperparameters, evaluate and gather statistics about an agent. See the [ExperimentManager](experimentManager_page) page.

### Seeding & Reproducibility
[rlberry](https://github.com/rlberry-py/rlberry) has a class [Seeder](rlberry.seeding.seeder.Seeder) that conveniently wraps a [NumPy SeedSequence](https://numpy.org/doc/stable/reference/random/parallel.html),
and allows us to create independent random number generators for different objects and threads, using a single
[Seeder](rlberry.seeding.seeder.Seeder) instance. See the [Seeding](seeding_page) page.


You can also save and load your experiments.
It could be useful in many way :
- don't repeat the training part every time.
- continue a previous training (or doing checkpoint).

See the [Save and Load Experiment](save_load_page) page.

### Statistical comparison of RL agents
The principal goal of rlberry is to give tools for proper experimentations in RL. In research, one of the usual tasks is to compare two or more RL agents, and for this one typically uses several seeds to train the agents several times and compare the resulting mean reward. We show here how to make sure that enough data and enough information were acquired to assert that two RL agents are indeed different. We propose two ways to do that: first are classical hypothesis testing and second are sequential testing scheme with AdaStop that aim at saving computation by stopping early if possible.
#### Compare agents
We give tools to compare several trained agents using the mean over a specify number of evaluations for each agent. The explanation can be found in the [user guide](comparison_page).

#### AdaStop
AdaStop is a Sequential testing for efficient and reliable comparison of stochastic algorithms. It has been successfully used to compare efficiently RL agents and an example of such use can be found in the [user guide](adastop_userguide).

[linked paper](https://hal-lara.archives-ouvertes.fr/hal-04132861/)


[GitHub](https://github.com/TimotheeMathieu/adastop)

### Visualization
TODO :


### And many more !
Check the [User Guide](user_guide) to find more tools !
