"""
=============================
UCB Bandit cumulative regret
=============================

This script shows how to define a bandit environment and an Index-based algorithm.
"""

import numpy as np
from rlberry.envs.bandits import NormalBandit
from rlberry.agents.bandits import IndexAgent, RecursiveIndexAgent
from rlberry.manager import AgentManager, plot_writer_data
import matplotlib.pyplot as plt
from rlberry.wrappers import WriterWrapper


# Agents definition


class UCBAgent(IndexAgent):
    """UCB agent for B-subgaussian bandits"""

    name = "UCB Agent"

    def __init__(self, env, B=1, **kwargs):
        def index(r, t):
            return np.mean(r) + B * np.sqrt(2 * np.log(t**2) / len(r))

        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


class RecursiveUCBAgent(RecursiveIndexAgent):
    """Same as above but defined recursively. Should give the same results (up to randomness)."""

    name = "Recursive UCB Agent"

    def __init__(self, env, B=1, **kwargs):
        def stat_function(stat, Na, action, reward):
            if stat is None:
                stat = np.zeros(len(Na))
            stat[action] = (Na[action] - 1) / Na[action] * stat[action] + reward / Na[
                action
            ]
            return stat

        def index(stat, Na, t):
            return stat + B * np.sqrt(2 * np.log(t**2) / Na)

        RecursiveIndexAgent.__init__(self, env, stat_function, index, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


class NaiveAgent(IndexAgent):
    name = "Naive Agent"

    def __init__(self, env, **kwargs):
        # choose the arm with the largest empirical mean
        IndexAgent.__init__(self, env, lambda r, t: np.mean(r), **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


# Parameters
means = [0, 0.9, 1]
T = 3000  # Horizon
M = 20  # number of MC simu


# Construction of the experiment


env_ctor = NormalBandit
env_kwargs = {"means": means, "stds": 2 * np.ones(len(means))}

agent1 = AgentManager(
    UCBAgent,
    (env_ctor, env_kwargs),
    fit_budget=T,
    init_kwargs={"B": 2},
    n_fit=M,
    parallelization="process",
    mp_context="fork",
)


agent1bis = AgentManager(
    RecursiveUCBAgent,
    (env_ctor, env_kwargs),
    fit_budget=T,
    init_kwargs={"B": 2},
    n_fit=M,
    parallelization="process",
    mp_context="fork",
)

agent2 = AgentManager(
    NaiveAgent,
    (env_ctor, env_kwargs),
    fit_budget=T,
    n_fit=M,
    parallelization="process",
    mp_context="fork",
)
# these parameters whould give parallel computing even in notebooks


# Agent training

agent1.fit()
agent1bis.fit()
agent2.fit()


# Compute and plot
def compute_regret(regret):
    return np.cumsum(np.max(means) - regret)


fig = plt.figure(1, figsize=(5, 3))
ax = plt.gca()
output = plot_writer_data(
    [agent1, agent1bis, agent2],
    tag="reward",
    preprocess_func=compute_regret,
    title="Cumulative Regret",
    ax=ax,
)
fig.subplots_adjust(bottom=0.2, left=0.2)
# you can customize the axis ax here to customize the plot
# or you can use output, which is a pandat DataFrame to do the plot yourself.
