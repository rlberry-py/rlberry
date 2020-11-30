from rlberry.agents import REINFORCEAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env


def test_reinforce_agent():
    env = get_benchmark_env(level=1)
    n_episodes = 5
    horizon = 30

    agent = REINFORCEAgent(env,
                     n_episodes=n_episodes,
                     horizon=horizon,
                     gamma=0.99,
                     learning_rate=0.001,
                     verbose=0)
    agent._log_interval = 0
    agent.fit()
    agent.policy(env.observation_space.sample())


def test_reinforce_agent_partial_fit():
    env = get_benchmark_env(level=1)
    n_episodes = 10
    horizon = 30

    agent = REINFORCEAgent(env,
                     n_episodes=n_episodes,
                     horizon=horizon,
                     gamma=0.99,
                     learning_rate=0.001,
                     verbose=1)
    agent._log_interval = 0

    agent.partial_fit(0.5)
    agent.policy(env.observation_space.sample())
    assert agent.episode == 5
    agent.partial_fit(0.5)
    assert agent.episode == 10
    agent.policy(env.observation_space.sample())
