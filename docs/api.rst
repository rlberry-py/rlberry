###########
rlberry API
###########

.. currentmodule:: rlberry

Manager
====================

Main classes
--------------------

.. autosummary::
  :toctree: generated/
  :template: class.rst


    manager.ExperimentManager
    manager.MultipleManagers

Evaluation and plot
--------------------

.. autosummary::
  :toctree: generated/
  :template: class.rst

   manager.AdastopComparator

.. autosummary::
   :toctree: generated/
   :template: function.rst

   manager.evaluate_agents
   manager.read_writer_data
   manager.plot_writer_data
   manager.plot_smoothed_curves
   manager.plot_synchronized_curves
   manager.compare_agents


Agents
====================

Base classes
------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    agents.Agent
    agents.AgentWithSimplePolicy


Agent importation tools
-----------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.stable_baselines.StableBaselinesAgent


Environments
============

Base class
----------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    envs.interface.Model
    envs.basewrapper.Wrapper

Spaces
------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    spaces.Discrete
    spaces.Box
    spaces.Tuple
    spaces.MultiDiscrete
    spaces.MultiBinary
    spaces.Dict


Environment tools
-----------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

    envs.gym_make
    envs.atari_make
    envs.PipelineEnv


Seeding
====================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   seeding.seeder.Seeder

.. autosummary::
   :toctree: generated/
   :template: function.rst

   seeding.safe_reseed
   seeding.set_external_seed

Utilities, Logging & Typing
===========================

Manager Utilitis
----------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   manager.preset_manager


Writer Utilities
----------------

.. autosummary::
  :toctree: generated/
  :template: class.rst

  utils.writers.DefaultWriter
  agents.utils.replay.ReplayBuffer

Check Utilities
---------------

 .. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.check_rl_agent
   utils.check_env
   utils.check_save_load
   utils.check_fit_additive
   utils.check_seeding_agent
   utils.check_experiment_manager

Logging Utilities
-----------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.logging.set_level


Benchmarks
----------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   benchmarks.benchmark_utils

Environment Wrappers
====================

.. autosummary::
  :toctree: generated/
  :template: class.rst

  wrappers.discretize_state.DiscretizeStateWrapper
  wrappers.gym_utils.OldGymCompatibilityWrapper
  wrappers.RescaleRewardWrapper
  wrappers.WriterWrapper
