"""
=============================
Record reward during training
=============================

This script shows how to modify an agent to easily record reward or action
during the fit of the agent.
"""


import numpy as np

from rlberry.wrappers import WriterWrapper
from rlberry.envs import GridWorld
from rlberry.manager import plot_writer_data, AgentManager
from rlberry.agents import UCBVIAgent

# We wrape the default writer of the agent in a WriterWrapper to record rewards.
class VIAgent(UCBVIAgent):
    name = 'UCBVIAgent'
    def __init__(self, env, **kwargs):
        UCBVIAgent.__init__(self, env, horizon = 50, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar = "reward")
        # we could also record actions with
        # self.env = WriterWrapper(self.env, self.writer, write_scalar = "action")


env_ctor = GridWorld
env_kwargs = dict(nrows=3, ncols=10,
                 reward_at = {(1,1):0.1, (2, 9):1.0},
                 walls=((1,4),(2,4), (1,5)),
                 success_probability=0.7)

env = env_ctor(**env_kwargs)
agent = AgentManager(VIAgent,
    (env_ctor, env_kwargs),
    fit_budget=10,
    n_fit=3)

agent.fit(budget=10)

# We use the following preprocessing function to plot the cumulative reward.
def compute_reward(rewards):
    return np.cumsum(rewards)

# Plot of the cumulative reward.
output = plot_writer_data(agent, tag="reward", preprocess_func=compute_reward, title="Cumulative Reward")
# The output is for 500 global steps because it uses 10 fit_budget * horizon
