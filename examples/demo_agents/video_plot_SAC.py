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


from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.agents.torch.a2c.a2c import A2CAgent
from rlberry.agents.torch import SACAgent

# We wrape the default writer of the agent in a WriterWrapper
# to record rewards.


class SSACAgent(SACAgent):
    name = "SACAgent"

    def __init__(self, env, **kwargs):
        SACAgent.__init__(self, env, horizon=100, batch_size=5, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


env_ctor = PBall2D
env_kwargs = dict()
env = env_ctor(**env_kwargs)

agent = AgentManager(SSACAgent, (env_ctor, env_kwargs), fit_budget=100, n_fit=3)
agent.fit()

# We use the following preprocessing function to plot the cumulative reward.
def compute_reward(rewards):
    return np.cumsum(rewards)


# Plot of the cumulative reward.
output = plot_writer_data(agent, tag="loss_q1", title="Loss q1")

output = plot_writer_data(agent, tag="loss_q2", title="Loss q2")

output = plot_writer_data(agent, tag="loss_v", title="Loss critic")

output = plot_writer_data(agent, tag="loss_act", title="Loss actor")
