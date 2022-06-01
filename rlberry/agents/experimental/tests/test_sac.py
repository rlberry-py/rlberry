from rlberry.agents.experimental.torch import SACAgent

from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.exploration_tools.discrete_counter import DiscreteCounter


def test_sac_agent():
    env = get_benchmark_env(level=1)
    n_episodes = 5

    def uncertainty_estimator_fn(observation_space, action_space):
        counter = DiscreteCounter(observation_space, action_space, n_bins_obs=20)
        return counter

    agent = SACAgent(
        env,
        gamma=0.99,
        learning_rate=0.001,
        k_epochs=4,
        use_bonus=True,
        uncertainty_estimator_kwargs=dict(
            uncertainty_estimator_fn=uncertainty_estimator_fn, bonus_scale_factor=1.0
        ),
        device="cpu",
    )
    agent.fit(budget=n_episodes)
    agent.policy(env.observation_space.sample())


def test_sac_agent_partial_fit():
    env = get_benchmark_env(level=1)
    n_episodes = 10

    agent = SACAgent(
        env,
        gamma=0.99,
        learning_rate=0.001,
        k_epochs=4,
        use_bonus=False,
        device="cpu",
    )

    agent.fit(budget=n_episodes // 2)
    agent.policy(env.observation_space.sample())
    assert agent.episode == 5
    agent.fit(budget=n_episodes // 2)
    assert agent.episode == 10
    agent.policy(env.observation_space.sample())
