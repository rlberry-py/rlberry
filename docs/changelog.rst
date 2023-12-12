.. _changelog:

Changelog
=========


Dev version
-----------

*PR #376*

 * New plot_writer_data function that does not depend on seaborn and that can plot smoothed function and confidence band if scikit-fda is installed.

Version 0.6.0
-------------

*PR #276*

 * Non adaptive multiple tests for agent comparison.

*PR #365*

 * Fix Sphinx version to <7.

*PR #350*

 * Rename AgentManager to ExperimentManager.

*PR #326*

 * Moved SAC from experimental to torch agents. Tested and benchmarked.

*PR #335*

 * Upgrade from Python3.9 -> python3.10


Version 0.5.0
-------------

*PR #281, #323*

 * Merge gymnasium branch into main, make gymnasium the default library for environments in rlberry.

Version 0.4.1
-------------

*PR #318*

* Update to allow the training on a computer with GPU, save the agents, then load it on a computer without GPU.

*PR #308*

* Update make_atari_env and PPO to be compatible together and use vectorized env (PPO manage the vector)

*PR #298*

* Move old scripts (jax agents, attention networks, old examples...) that we won't maintain from the main branch to an archive branch.

*PR #277*

* Add and update code to use "Atari games" env

*PR #281*

* New branch for code compatible with Gymnasium

Version 0.4.0
-------------

*PR #273*

* Change the default behavior of `plot_writer_data` so that if seaborn has version >= 0.12.0 then
  a 90% percentile interval is used instead of sd.

*PR #269*

* Add :class:`rlberry.envs.PipelineEnv` a way to define pipeline of wrappers in a simple way.

*PR #262*

* PPO can now handle continuous actions.

*PR #261, #264*

* Implementation of Munchausen DQN in :class:`rlberry.agents.torch.MDQNAgent`.
* Comparison of MDQN with DQN agent in the long tests.


*PR #244, #250, #253*

* Compress the pickles used to save the trained agents.

*PR #235*

* Implementation  of :class:`rlberry.envs.SpringCartPole` environment, an RL environment featuring two cartpoles linked by a spring.

*PR #226, #227*

* Improve logging, the logging level can now be changed with :py:func:`rlberry.utils.logging.set_level`.
* Introduce smoothing in curves done with `plot_writer_data` when only one seed is used.


*PR #223*

* Moved PPO from experimental to torch agents. Tested and benchmarked.


Version 0.3.0
-------------


*PR #206*

* Creation of a Deep RL tutorial, in the user guide.

*PR #132*

* New tracker class :class:`rlberry.agents.bandit.tools.BanditTracker` to track statistics to be used in Bandit algorithms.

*PR #191*

* Possibility to generate a profile with :class:`rlberry.agents.manager.ExperimentManager`.

*PR #148, #161, #180*

* Misc improvements on A2C.
* New StableBaselines3 wrapper :class:`rlberry.agents.stable_baselines.StableBaselinesAgent` to import StableBaselines3 Agents.

*PR #119*

* Improving documentation for agents.torch.utils
* New replay buffer :class:`rlberry.agents.utils.replay.ReplayBuffer`, aiming to replace code in utils/memories.py
* New DQN implementation, aiming to fix reproducibility and compatibility issues.
* Implements Q(lambda) in DQN Agent.


*Feb 22, 2022 (PR #126)*

* Setup :code:`rlberry.__version__` (currently 0.3.0dev0)
* Record rlberry version in a ExperimentManager attribute equality of ExperimentManagers
* Override :code:`__eq__` method of the ExperimentManager class.

*Feb 14-15, 2022 (PR #97, #118)*

* (feat) Add Bandits basic environments and agents. See :class:`~rlberry.agents.bandits.IndexAgent` and :class:`~rlberry.envs.bandits.Bandit`.
* Thompson Sampling bandit algorithm with gaussian or beta prior.
* Base class for bandits algorithms with custom save & load functions (called :class:`~rlberry.agents.bandits.BanditWithSimplePolicy`)


*Feb 11, 2022 (#83, #95)*

* (fix) Fixed bug in :meth:`FiniteMDP.sample()`: terminal state was being checked with `self.state` instead of given `state`
* (feat) Option to use 'fork' or 'spawn' in :class:`~rlberry.manager.ExperimentManager`
* (feat) ExperimentManager output_dir now has a timestamp and a short ID by default.
* (feat) Gridworld can be constructed from string layout
* (feat) `max_workers` argument for :class:`~rlberry.manager.ExperimentManager` to control the maximum number of processes/threads created by the :meth:`fit` method.


*Feb 04, 2022*

* Add :class:`~rlberry.manager.read_writer_data` to load agent's writer data from pickle files and make it simpler to customize in :class:`~rlberry.manager.plot_writer_data`
* Fix bug, dqn should take a tuple as environment
* Add a quickstart tutorial in the docs :ref:`quick_start`
* Add the RLSVI algorithm (tabular) :class:`~rlberry.agents.RLSVIAgent`
* Add the Posterior Sampling for Reinforcement Learning PSRL agent for tabular MDP :class:`~rlberry.agents.PSRLAgent`
* Add a page to help contributors in the doc :ref:`contributing`

Version 0.2.1
-------------


* :class:`~rlberry.agents.Agent` and :class:`~rlberry.manager.ExperimentManager` both have a unique_id attribute (useful for creating unique output files/directories).
* `DefaultWriter` is now initialized in base class `Agent` and (optionally) wraps a tensorboard `SummaryWriter`.
* :class:`~rlberry.manager.ExperimentManager` has an option enable_tensorboard that activates tensorboard logging in each of its Agents (with their writer attribute). The log_dirs of tensorboard are automatically assigned by :class:`~rlberry.manager.ExperimentManager`.
* `RemoteExperimentManager` receives tensorboard data created in the server, when the method `get_writer_data()` is called. This is done by a zip file transfer with :class:`~rlberry.network`.
* `BaseWrapper` and `gym_make` now have an option `wrap_spaces`. If set to `True`, this option converts `gym.spaces` to `rlberry.spaces`, which provides classes with better seeding (using numpy's default_rng instead of `RandomState`)
* :class:`~rlberry.manager.ExperimentManager`: new method `get_agent_instances()` that returns trained instances
* `plot_writer_data`: possibility to set `xtag` (tag used for x-axis)
* Fixed agent initialization bug in `AgentHandler` (`eval_env` missing in `kwargs` for agent_class).


Version 0.2
-----------

* `AgentStats` renamed to :class:`~rlberry.manager.ExperimentManager`.
* :class:`~rlberry.manager.ExperimentManager` can handle agents that cannot be pickled.
* Agent interface requires `eval()` method instead of `policy()` to handle more general agents (e.g. reward-free, POMDPs etc).
* Multi-processing and multi-threading are now done with `ProcessPoolExecutor` and `ThreadPoolExecutor` (allowing nested processes for example). Processes are created with spawn (jax does not work with fork, see #51).
* JAX implementation of DQN and replay buffer using reverb (experimental).
* :class:`~rlberry.network`: server and client interfaces to exchange messages via sockets (experimental).
* `RemoteExperimentManager` to train agents in a remote server and gather the results locally (experimental).
* Fix rendering bug with OpenGL
