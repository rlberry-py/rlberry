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


Torch Agents
--------------------

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

Still experimental. See source code rlberry.agents.jax for more info.


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
.. autosummary::
  :toctree: generated/
  :template: class.rst

  utils.writers.DefaultWriter


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
