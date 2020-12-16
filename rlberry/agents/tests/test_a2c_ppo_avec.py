from copy import deepcopy
from rlberry.agents import A2CAgent
from rlberry.agents import PPOAgent
from rlberry.agents import AVECPPOAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.wrappers.uncertainty_estimator_wrapper import \
    UncertaintyEstimatorWrapper


def test_a2c_agent():
    env = get_benchmark_env(level=1)
    n_episodes = 5
    horizon = 30

    def uncertainty_estimator_fn(observation_space, action_space):
        counter = DiscreteCounter(observation_space,
                                  action_space,
                                  n_bins_obs=20)
        return counter

    agent = A2CAgent(env,
                     n_episodes=n_episodes,
                     horizon=horizon,
                     gamma=0.99,
                     learning_rate=0.001,
                     k_epochs=4,
                     use_bonus=True,
                     uncertainty_estimator_kwargs=dict(
                         uncertainty_estimator_fn=uncertainty_estimator_fn,
                         bonus_scale_factor=1.0
                     ))
    agent._log_interval = 0
    agent.fit()
    agent.policy(env.observation_space.sample())


def test_a2c_agent_partial_fit():
    env = get_benchmark_env(level=1)
    n_episodes = 10
    horizon = 30

    agent = A2CAgent(env,
                     n_episodes=n_episodes,
                     horizon=horizon,
                     gamma=0.99,
                     learning_rate=0.001,
                     k_epochs=4,
                     use_bonus=False)
    agent._log_interval = 0

    agent.partial_fit(0.5)
    agent.policy(env.observation_space.sample())
    assert agent.episode == 5
    agent.partial_fit(0.5)
    assert agent.episode == 10
    agent.policy(env.observation_space.sample())


def test_ppo_agent():
    env = get_benchmark_env(level=1)
    n_episodes = 5
    horizon = 30

    def uncertainty_estimator_fn(observation_space, action_space):
        counter = DiscreteCounter(observation_space,
                                  action_space,
                                  n_bins_obs=20)
        return counter

    agent = PPOAgent(env,
                     n_episodes=n_episodes,
                     horizon=horizon,
                     gamma=0.99,
                     learning_rate=0.001,
                     eps_clip=0.2,
                     k_epochs=4,
                     use_bonus=True,
                     uncertainty_estimator_kwargs=dict(
                         uncertainty_estimator_fn=uncertainty_estimator_fn,
                         bonus_scale_factor=1
                     ))
    agent._log_interval = 0
    agent.fit()
    agent.policy(env.observation_space.sample())


def test_ppo_agent_partial_fit():
    env = get_benchmark_env(level=1)
    n_episodes = 10
    horizon = 30

    agent = PPOAgent(env,
                     n_episodes=n_episodes,
                     horizon=horizon,
                     gamma=0.99,
                     learning_rate=0.001,
                     eps_clip=0.2,
                     k_epochs=4,
                     batch_size=1,
                     use_bonus=False)
    agent._log_interval = 0

    agent.partial_fit(0.5)
    agent.policy(env.observation_space.sample())
    assert agent.episode == 5
    agent.partial_fit(0.5)
    assert agent.episode == 10
    agent.policy(env.observation_space.sample())


def test_avec_ppo_agent():
    env = get_benchmark_env(level=1)
    n_episodes = 5
    horizon = 30

    #
    def uncertainty_estimator_fn(observation_space, action_space):
        counter = DiscreteCounter(observation_space,
                                  action_space,
                                  n_bins_obs=20)
        return counter

    agent = AVECPPOAgent(env,
                         n_episodes=n_episodes,
                         horizon=horizon,
                         gamma=0.99,
                         learning_rate=0.001,
                         eps_clip=0.2,
                         k_epochs=4,
                         batch_size=1,
                         use_bonus=True,
                         uncertainty_estimator_kwargs=dict(
                             uncertainty_estimator_fn=uncertainty_estimator_fn,
                             bonus_scale_factor=1.0)
                         )
    agent._log_interval = 0
    agent.fit()
    agent.policy(env.observation_space.sample())


def test_avec_ppo_agent_partial_fit():
    env = get_benchmark_env(level=1)
    n_episodes = 10
    horizon = 30

    agent = AVECPPOAgent(env,
                         n_episodes=n_episodes,
                         horizon=horizon,
                         gamma=0.99,
                         learning_rate=0.001,
                         eps_clip=0.2,
                         k_epochs=4,
                         batch_size=1,
                         use_bonus=False)
    agent._log_interval = 0

    agent.partial_fit(0.5)
    agent.policy(env.observation_space.sample())
    assert agent.episode == 5
    agent.partial_fit(0.5)
    assert agent.episode == 10
    agent.policy(env.observation_space.sample())
