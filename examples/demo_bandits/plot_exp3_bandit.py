"""
=============================
EXP3 Bandit cumulative regret
=============================

This script shows how to define a bandit environment and an EXP3
randomized algorithm.
"""

import numpy as np
from rlberry.envs.bandits import BernoulliBandit
from rlberry.agents.bandits import RandomizedAgent
from rlberry.manager import AgentManager, plot_writer_data
import matplotlib.pyplot as plt
from rlberry.wrappers import WriterWrapper


# Agents definition


class EXP3Agent(RandomizedAgent):
    name = "EXP3"

    def __init__(self, env, **kwargs):
        def index(r, p, t):
            return np.sum(1 - (1 - r) / p)

        def prob(indices, t):
            eta = np.minimum(
                np.sqrt(self.n_arms * np.log(self.n_arms) / (t + 1)),
                1.
            )
            w = np.exp(eta * indices)
            w /= w.sum()
            return (1 - eta) * w + eta * np.ones(self.n_arms) / self.n_arms

        RandomizedAgent.__init__(self, env, index, prob, **kwargs)
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="reward"
        )


# Parameters of the problem
p = [0.4, 0.5, 0.6]  # means of the arms
T = 3000  # Horizon
M = 20  # number of MC simu

# Construction of the experiment

env_ctor = BernoulliBandit
env_kwargs = {"p": p}

agent = AgentManager(
    EXP3Agent,
    (env_ctor, env_kwargs),
    fit_budget=T,
    init_kwargs={},
    n_fit=M,
    parallelization="process",
    mp_context="fork",
)
# these parameters should give parallel computing even in notebooks


# Agent training

agent.fit()


# Compute and plot regret
def compute_regret(regret):
    return np.cumsum(np.max(p) - regret)


fig = plt.figure(1, figsize=(5, 3))
ax = plt.gca()
output = plot_writer_data(
    [agent],
    tag="reward",
    preprocess_func=compute_regret,
    title="Cumulative Regret",
    ax=ax,
)
