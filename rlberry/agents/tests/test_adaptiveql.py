from rlberry.agents.adaptiveql import AdaptiveQLAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
import matplotlib.pyplot as plt


def test_adaptive_ql():
    env = get_benchmark_env(level=2)
    agent = AdaptiveQLAgent(env, n_episodes=50, horizon=30)
    agent.fit()
    agent.policy(env.observation_space.sample())
    agent.Qtree.plot(0, 20)
    plt.clf()
