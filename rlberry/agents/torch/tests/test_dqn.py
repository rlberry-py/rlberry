from rlberry.envs import gym_make
from rlberry.agents.torch.dqn import DQNAgent
from rlberry.seeding import Seeder
import numpy as np


def test_dqn_agent():
    env = gym_make("MountainCar-v0")
    agent = DQNAgent(env)
    agent.fit(budget=100)
