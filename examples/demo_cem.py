import numpy as np
from rlberry.agents.cem import CEMAgent
from rlberry.envs.toy_exploration import PBall2D

import rlberry.seeding as seeding 

seeding.set_global_seed(123)

env = PBall2D(p=np.inf, reward_smoothness=np.array([0.8]), reward_centers=[np.array([0.4, 0.4])])
n_episodes  = 500
batch_size = 100
horizon = 25
gamma = 0.99

agent = CEMAgent(env, n_episodes, horizon, gamma, batch_size, percentile=70, learning_rate=0.01)
agent.fit()
   
env.enable_rendering()
state = env.reset()
for tt in range(4*horizon):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

env.render()
