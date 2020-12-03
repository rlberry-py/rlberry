import numpy as np
from rlberry.agents.cem import CEMAgent
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
import rlberry.seeding as seeding

seeding.set_global_seed(1234)

env = get_benchmark_env(level=1)

n_episodes = 700
batch_size = 50
horizon = 30
gamma = 0.99

agent = CEMAgent(env, n_episodes, horizon, gamma, batch_size,
                 percentile=70, learning_rate=0.01)
agent.fit()

env.enable_rendering()
state = env.reset()
for tt in range(4*horizon):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

env.render()
