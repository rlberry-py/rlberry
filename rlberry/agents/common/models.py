import torch
from torch import nn as nn
from torch.distributions import Categorical


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(ValueNet, self).__init__()
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        state_value = self.critic(state)
        return torch.squeeze(state_value)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyNet, self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        action_probs = self.softmax(self.actor(state))
        dist = Categorical(action_probs)
        return dist

    def action_scores(self, state):
        action_scores = self.actor(state)
        return action_scores