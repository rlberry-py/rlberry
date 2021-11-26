###########
rlberry API
###########

.. currentmodule:: rlberry

Agents
====================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.agent.AgentWithSimplePolicy
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
   agents.kernel_based.kernels.kernel_func
   agents.linear.LSVIUCBAgent
   agents.torch.a2c.A2CAgent
   agents.torch.ppo.PPOAgent
   agents.torch.AVECPPOAgent
   agents.torch.REINFORCEAgent


Manager
====================

   .. autosummary::
      :toctree: generated/
      :template: class.rst


    manager.AgentManager
    manager.evaluate_agents
    manager.MultipleManagers
    manager.plot_writer_data


Environments
====================

.. autosummary::
   :toctree: generated/
   :template: class.rst

    envs.Acrobot
    envs.gym_make
    envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
    envs.benchmarks.ball_exploration.PBall2D
    envs.benchmarks.generalization.twinrooms.TwinRooms
    envs.benchmarks.grid_exploration.apple_gold.AppleGold
    envs.benchmarks.grid_exploration.nroom.NRoom
    envs.classic_control.MountainCar
    envs.finite.Chain
    envs.finite.GridWorld


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
