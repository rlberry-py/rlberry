from rlberry.agents.adaptiveql import AdaptiveQLAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
import matplotlib.pyplot as plt

env = get_benchmark_env(level=2)
agent = AdaptiveQLAgent(env, n_episodes=20000, horizon=30, gamma=0.99)
agent.fit()
agent.policy(env.observation_space.sample())
agent.Qtree.plot(0, 25)
plt.show()
