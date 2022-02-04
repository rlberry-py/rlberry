.. _changelog:

Changelog
=========

Version 0.2.0
-------------

*Feb 04, 2022*

* Add :class:`~rlberry.manager.read_writer_data` to load agent's writer data from pickle files and make it simpler to customize in :class:`~rlberry.manager.plot_writer_data`
* Fix bug, dqn should take a tuple as environment
* Add a quickstart tutorial in the docs :ref:`quick_start`
* Add the RLSVI algorithm (tabular) :class:`~rlberry.agents.RLSVIAgent`
* Add the Posterior Sampling for Reinforcement Learning PSRL agent for tabular MDP :class:`~rlberry.agents.PSRLAgent`
* Add a page to help contributors in the doc :ref:`contributing`
