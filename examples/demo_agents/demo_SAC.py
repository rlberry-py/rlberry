"""
=============================
Record reward during training
=============================

This script shows how to modify an agent to easily record reward or action
during the fit of the agent.
"""


# import numpy as np
# from rlberry.wrappers import WriterWrapper
from rlberry.envs.basewrapper import Wrapper

# from rlberry.envs import gym_make
from rlberry.manager import plot_writer_data, AgentManager
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents.experimental.torch import SACAgent
import gym


# we dont need wrapper actually just 'return env' works
def env_ctor(env, wrap_spaces=True):
    return Wrapper(env, wrap_spaces)


env = PBall2D()
env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
env_kwargs = dict(env=env)
agent = AgentManager(
    SACAgent,
    (env_ctor, env_kwargs),
    fit_budget=500,
    n_fit=1,
    enable_tensorboard=True,
)

# basic version
# env_kwargs = dict(id = "CartPole-v0")
# agent = AgentManager(SACAgent, (gym_make, env_kwargs), fit_budget=200, n_fit=1)

# # timothe's
# env = gym_make("CartPole-v0")
# agent = AgentManager(
#     SACAgent, (env.__class__, dict()), fit_budget=200, n_fit=1, enable_tensorboard=True,
# )

# Omar's
# env = gym_make("CartPole-v0")
# from copy import deepcopy
# def env_constructor():
#     return deepcopy(env)
# agent = AgentManager(
#     SACAgent, (env_constructor, dict()), fit_budget=200, n_fit=1, enable_tensorboard=True,
# )


agent.fit()

# Plot of the cumulative reward.
output = plot_writer_data(agent, tag="loss_q1", title="Loss q1")
output = plot_writer_data(agent, tag="loss_q2", title="Loss q2")
output = plot_writer_data(agent, tag="loss_v", title="Loss critic")
output = plot_writer_data(agent, tag="loss_act", title="Loss actor")
