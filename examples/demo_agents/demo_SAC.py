"""
=============================
SAC Soft Actor-Critic
=============================

This script shows how to train a SAC agent on a Pendulum environment.
"""

import time

import gymnasium as gym
from rlberry.agents.torch.sac import SACAgent
from rlberry.envs import Pendulum
from rlberry.manager import AgentManager


def env_ctor(env, wrap_spaces=True):
    return env


# Setup agent parameters
env_name = "Pendulum"
fit_budget = int(2e5)
agent_name = f"{env_name}_{fit_budget}_{int(time.time())}"

# Setup environment parameters
env = Pendulum()
env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
env = gym.wrappers.RecordEpisodeStatistics(env)
env_kwargs = dict(env=env)

# Create agent instance
agent = AgentManager(
    SACAgent,
    (env_ctor, env_kwargs),
    fit_budget=fit_budget,
    n_fit=1,
    enable_tensorboard=True,
    agent_name=agent_name,
)

# Start training
agent.fit()
