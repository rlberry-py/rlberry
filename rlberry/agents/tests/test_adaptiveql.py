from rlberry.agents import AdaptiveQLAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
import matplotlib.pyplot as plt


def test_adaptive_ql():
    env = get_benchmark_env(level=2)
    agent = AdaptiveQLAgent(env, horizon=30)
    agent.fit(budget=50)
    agent.policy(env.observation_space.sample())
    agent.Qtree.plot(0, 20)
    plt.clf()
