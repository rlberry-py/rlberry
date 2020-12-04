from rlberry.agents.cem import CEMAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env


def test_cem_agent():
    env = get_benchmark_env(level=1)
    n_episodes = 5
    batch_size = 100
    horizon = 30
    gamma = 0.99

    agent = CEMAgent(env,
                     n_episodes,
                     horizon=horizon,
                     gamma=gamma,
                     batch_size=batch_size,
                     percentile=70,
                     learning_rate=0.01)
    agent._log_interval = 0
    agent.fit()
    agent.policy(env.observation_space.sample())


def test_cem_agent_partial_fit():
    env = get_benchmark_env(level=1)
    n_episodes = 10
    batch_size = 100
    horizon = 30
    gamma = 0.99

    agent = CEMAgent(env,
                     n_episodes,
                     horizon=horizon,
                     gamma=gamma,
                     batch_size=batch_size,
                     percentile=70,
                     learning_rate=0.01)

    agent._log_interval = 0

    agent.partial_fit(0.5)
    agent.policy(env.observation_space.sample())
    assert agent.episode == 5
    agent.partial_fit(0.5)
    assert agent.episode == 10
    agent.policy(env.observation_space.sample())
