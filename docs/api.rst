###########
rlberry API
###########

.. currentmodule:: rlberry

Agents
====================

Interface
----------
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
   agents.linear.LSVIUCBAgent
   agents.RLSVIAgent
   agents.PSRLAgent

Bandits
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.bandits.IndexAgent
   agents.bandits.RecursiveIndexAgent

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
====================

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


Bandits
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   envs.bandits.Bandit
   envs.bandits.NormalBandit
   envs.bandits.CorruptedNormalBandit

Base class
----------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    envs.interface.Model


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


Utilities & Logging
====================

Writer Utilities
----------------

.. autosummary::
  :toctree: generated/
  :template: class.rst

  utils.writers.DefaultWriter


Check Utilities
---------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.check_bandit_agent



Wrappers
====================

.. autosummary::
  :toctree: generated/
  :template: class.rst

  wrappers.discretize_state.DiscretizeStateWrapper
  wrappers.RescaleRewardWrapper
  wrappers.scalarize.ScalarizeEnvWrapper
  wrappers.vis2d.Vis2dWrapper
  wrappers.WriterWrapper
