""" 
 ===================== 
 Demo: demo_pybullet 
 =====================
"""
from torch.utils.tensorboard import SummaryWriter

import rlberry.envs.bullet3.pybullet_envs
import space_wrappers
import gym

from rlberry.agents.torch.dqn import DQNAgent

env = gym.make("DiscretePendulumSwingupBulletEnv-v0")

env.render()
agent = DQNAgent(env, gamma=0.95, learning_rate=0.0005, copy_env=False)
agent.set_writer(SummaryWriter())
agent.fit(budget=100)

while True:
    state = env.reset()
    done = False
    while not done:
        action = agent.policy(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
