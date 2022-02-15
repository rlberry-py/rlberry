"""
 =====================
 Demo: dqn_nets
 =====================
 Neural nets for the DQN agent.
"""


import gym.spaces
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, env):
        super(QNet, self).__init__()
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.fc1 = nn.Linear(obs_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, observation):
        x = self.relu(self.fc1(observation))
        x = self.relu(self.fc2(x))
        qvals = self.fc3(x)
        return qvals
