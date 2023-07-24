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


    manager.AgentManager
    manager.MultipleManagers

Evaluation and plot
--------------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   manager.evaluate_agents
   manager.read_writer_data
   manager.plot_writer_data


Agents
====================

Base classes
------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    agents.Agent
    agents.AgentWithSimplePolicy

Basic Agents
--------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.QLAgent
   agents.SARSAAgent
   agents.ValueIterationAgent
   agents.MBQVIAgent
   agents.UCBVIAgent
   agents.RSUCBVIAgent
   agents.RSKernelUCBVIAgent
   agents.OptQLAgent
   agents.LSVIUCBAgent
   agents.RLSVIAgent
   agents.PSRLAgent


Agent importation tools
-----------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.stable_baselines.StableBaselinesAgent


Torch Agents
---------------------------


.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.torch.SACAgent
   agents.torch.A2CAgent
   agents.torch.PPOAgent
   agents.torch.DQNAgent
   agents.torch.MunchausenDQNAgent
   agents.torch.REINFORCEAgent


Environments
============

Base class
----------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    envs.interface.Model

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

Benchmark Environments
----------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

    envs.Acrobot
    envs.benchmarks.ball_exploration.PBall2D
    envs.benchmarks.generalization.twinrooms.TwinRooms
    envs.benchmarks.grid_exploration.apple_gold.AppleGold
    envs.benchmarks.grid_exploration.nroom.NRoom
    envs.classic_control.MountainCar
    envs.SpringCartPole
    envs.finite.Chain
    envs.finite.GridWorld


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
   utils.check_agent_manager

Logging Utilities
-----------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.logging.set_level


Typing
------

.. autosummary::
  :toctree: generated/
  :template: class.rst

   types.Env


Environment Wrappers
====================

.. autosummary::
  :toctree: generated/
  :template: class.rst

  wrappers.discretize_state.DiscretizeStateWrapper
  wrappers.gym_utils.OldGymCompatibilityWrapper
  wrappers.RescaleRewardWrapper
  wrappers.vis2d.Vis2dWrapper
  wrappers.WriterWrapper


Neural Networks
===============


Torch
------

.. autosummary::
  :toctree: generated/
  :template: function.rst

  agents.torch.utils.training.model_factory
  utils.torch.choose_device


.. autosummary::
  :toctree: generated/
  :template: class.rst

  agents.torch.utils.models.MultiLayerPerceptron
  agents.torch.utils.models.ConvolutionalNetwork
  agents.torch.utils.models.DuelingNetwork
  agents.torch.utils.models.Table


Bandits
=======

Bandit environments
-------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   envs.bandits.AdversarialBandit
   envs.bandits.Bandit
   envs.bandits.BernoulliBandit
   envs.bandits.NormalBandit
   envs.bandits.CorruptedNormalBandit

Bandit algorithms
-----------------
The bandits algorithms use mainly the following tracker tool:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.bandits.tools.BanditTracker

Some general class of bandit algorithms are provided.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.bandits.BanditWithSimplePolicy
   agents.bandits.IndexAgent
   agents.bandits.RandomizedAgent
   agents.bandits.TSAgent

A number of indices are provided to use in bandits algorithms:

.. autosummary::
  :toctree: generated/
  :template: function.rst

  agents.bandits.makeBoundedIMEDIndex
  agents.bandits.makeBoundedMOSSIndex
  agents.bandits.makeBoundedNPTSIndex
  agents.bandits.makeBoundedUCBIndex
  agents.bandits.makeBoundedUCBVIndex
  agents.bandits.makeETCIndex
  agents.bandits.makeEXP3Index
  agents.bandits.makeSubgaussianMOSSIndex
  agents.bandits.makeSubgaussianUCBIndex
