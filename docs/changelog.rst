.. _changelog:

Changelog
=========

Under development
-----------------

*Feb 04, 2022*

* Add :class:`~rlberry.manager.read_writer_data` to load agent's writer data from pickle files and make it simpler to customize in :class:`~rlberry.manager.plot_writer_data`
* Fix bug, dqn should take a tuple as environment
* Add a quickstart tutorial in the docs :ref:`quick_start`
* Add the RLSVI algorithm (tabular) :class:`~rlberry.agents.RLSVIAgent`
* Add the Posterior Sampling for Reinforcement Learning PSRL agent for tabular MDP :class:`~rlberry.agents.PSRLAgent`
* Add a page to help contributors in the doc :ref:`contributing`

Version 0.2.1 (last released version)
-------------------------------------


* :class:`~rlberry.agents.Agent` and :class:`~rlberry.manager.AgentManager` both have a unique_id attribute (useful for creating unique output files/directories).
* `DefaultWriter` is now initialized in base class `Agent` and (optionally) wraps a tensorboard `SummaryWriter`.
* :class:`~rlberry.manager.AgentManager` has an option enable_tensorboard that activates tensorboard logging in each of its Agents (with their writer attribute). The log_dirs of tensorboard are automatically assigned by :class:`~rlberry.manager.AgentManager`.
* `RemoteAgentManager` receives tensorboard data created in the server, when the method `get_writer_data()` is called. This is done by a zip file transfer with :class:`~rlberry.network`.
* `BaseWrapper` and `gym_make` now have an option `wrap_spaces`. If set to `True`, this option converts `gym.spaces` to `rlberry.spaces`, which provides classes with better seeding (using numpy's default_rng instead of `RandomState`)
* :class:`~rlberry.manager.AgentManager`: new method `get_agent_instances()` that returns trained instances
* `plot_writer_data`: possibility to set `xtag` (tag used for x-axis)
* Fixed agent initialization bug in `AgentHandler` (`eval_env` missing in `kwargs` for agent_class).


Version 0.2
-----------

* `AgentStats` renamed to :class:`~rlberry.manager.AgentManager`.
* :class:`~rlberry.manager.AgentManager` can handle agents that cannot be pickled.
* Agent interface requires `eval()` method instead of `policy()` to handle more general agents (e.g. reward-free, POMDPs etc).
* Multi-processing and multi-threading are now done with `ProcessPoolExecutor` and `ThreadPoolExecutor` (allowing nested processes for example). Processes are created with spawn (jax does not work with fork, see #51).
* JAX implementation of DQN and replay buffer using reverb (experimental).
* :class:`~rlberry.network`: server and client interfaces to exchange messages via sockets (experimental).
* `RemoteAgentManager` to train agents in a remote server and gather the results locally (experimental).
* Fix rendering bug with OpenGL
