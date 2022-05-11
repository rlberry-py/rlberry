###########
rlberry API
###########

.. currentmodule:: rlberry


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

   agents.ValueIterationAgent
   agents.MBQVIAgent
   agents.UCBVIAgent
   agents.RSUCBVIAgent
   agents.RSKernelUCBVIAgent
   agents.OptQLAgent
   agents.LSVIUCBAgent
   agents.RLSVIAgent
   agents.PSRLAgent


 Importation tools
-----------------------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.stable_baselines.StableBaselinesAgent


Torch Agents (experimental)
---------------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.torch.DQNAgent
   agents.torch.A2CAgent
   agents.torch.PPOAgent
   agents.torch.AVECPPOAgent
   agents.torch.REINFORCEAgent

Jax Agents (experimental)
--------------------

Still experimental. Look at the source of `rlberry.agents.jax` for more info.

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


Environments
============

Base class
----------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    envs.interface.Model

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
    envs.finite.Chain
    envs.finite.GridWorld


Importation tools
-----------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

    envs.gym_make


Seeding
====================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   seeding.seeder.Seeder


Utilities, Logging & Typing
====================

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
  wrappers.RescaleRewardWrapper
  wrappers.scalarize.ScalarizeEnvWrapper
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
  agents.torch.utils.attention_models.EgoAttentionNetwork


Bandits
=======

Environments
------------

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

.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.bandits.BanditWithSimplePolicy
   agents.bandits.IndexAgent
   agents.bandits.RandomizedAgent
   agents.bandits.TSAgent
