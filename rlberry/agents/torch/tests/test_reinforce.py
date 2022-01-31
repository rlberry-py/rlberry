from rlberry.agents.torch import REINFORCEAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.wrappers.uncertainty_estimator_wrapper import UncertaintyEstimatorWrapper


def test_reinforce_agent():
    _env = get_benchmark_env(level=1)
    n_episodes = 50
    horizon = 30

    #
    def uncertainty_estimator_fn(observation_space, action_space):
        counter = DiscreteCounter(observation_space, action_space, n_bins_obs=20)
        return counter

    env = UncertaintyEstimatorWrapper(
        _env, uncertainty_estimator_fn, bonus_scale_factor=1.0
    )
    #
    agent = REINFORCEAgent(
        env,
        horizon=horizon,
        gamma=0.99,
        learning_rate=0.001,
        use_bonus_if_available=True,
    )
    agent.fit(budget=n_episodes)
    agent.policy(env.observation_space.sample())


def test_reinforce_agent_partial_fit():
    env = get_benchmark_env(level=1)
    n_episodes = 10
    horizon = 30

    agent = REINFORCEAgent(
        env,
        horizon=horizon,
        gamma=0.99,
        learning_rate=0.001,
        use_bonus_if_available=False,
    )
    agent.fit(budget=n_episodes // 2)
    agent.policy(env.observation_space.sample())
    assert agent.episode == 5
    agent.fit(budget=n_episodes // 2)
    assert agent.episode == 10
    agent.policy(env.observation_space.sample())
