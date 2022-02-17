"""
=============================================================
Comparison of Thompson sampling and UCB on Bernoulli bandits.
=============================================================

This script shows how to use Thompson sampling.
"""

import numpy as np
from rlberry.envs.bandits import BernoulliBandit
from rlberry.agents.bandits import TSAgent, RecursiveIndexAgent
from rlberry.manager import AgentManager, plot_writer_data
from rlberry.wrappers import WriterWrapper


# Agents definition


class Thompson_samplingAgent(TSAgent):
    """Thompson_ sampling for bernoulli rvs"""

    name = "Thompson sampling"

    def __init__(self, env, B=1, **kwargs):
        TSAgent.__init__(self, env, "beta", **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="action")


class UCBAgent(RecursiveIndexAgent):
    name = "UCB"

    def __init__(self, env, **kwargs):
        RecursiveIndexAgent.__init__(
            self, env, **kwargs
        )  # default is UCB for Bernoulli.
        self.env = WriterWrapper(self.env, self.writer, write_scalar="action")


# Parameters of the problem
means = np.array([0.8, 0.8, 0.9, 1])  # means of the arms
A = len(means)
T = 2000  # Horizon
M = 10  # number of MC simu

# Construction of the experiment

env_ctor = BernoulliBandit
env_kwargs = {"p": means}

agents = [
    AgentManager(
        Agent,
        (env_ctor, env_kwargs),
        fit_budget=T,
        n_fit=M,
    )
    for Agent in [UCBAgent, Thompson_samplingAgent]
]

# Agent training

for agent in agents:
    agent.fit()


# Compute and plot (pseudo-)regret
def compute_pseudo_regret(action):
    return np.cumsum(np.max(means) - means[action.astype(int)])


output = plot_writer_data(
    agents,
    tag="action",
    preprocess_func=compute_pseudo_regret,
    title="Cumulative Pseudo-Regret",
)
