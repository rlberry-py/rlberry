###########
rlberry API
###########

.. currentmodule:: rlberry

Agents
====================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.AgentWithSimplePolicy
   agents.features.FeatureMap
   agents.ValueIterationAgent
   agents.MBQVIAgent
   agents.UCBVIAgent
   agents.RSUCBVIAgent
   agents.RSKernelUCBVIAgent
   agents.optql.OptQLAgent
   agents.MBQVIAgent
   agents.MBQVIAgent
   agents.torch.dqn.DQNAgent
   agents.linear.LSVIUCBAgent
   agents.torch.a2c.A2CAgent
   agents.torch.ppo.PPOAgent
   agents.torch.AVECPPOAgent
   agents.torch.REINFORCEAgent


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

    envs.gym_make.gym_make
    envs.gym_make.atari_make



Seeding
====================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   seeding.seeder.Seeder

Wrappers
====================

.. autosummary::
  :toctree: generated/
  :template: class.rst

  wrappers.discretize_state.DiscretizeStateWrapper
  wrappers.RescaleRewardWrapper
  wrappers.scalarize.ScalarizeEnvWrapper
  wrappers.vis2d.Vis2dWrapper
