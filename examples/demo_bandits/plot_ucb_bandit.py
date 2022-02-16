"""
=============================
UCB Bandit cumulative regret
=============================

This script shows how to define a bandit environment and an UCB Index-based algorithm.
"""

import numpy as np
from rlberry.envs.bandits import NormalBandit
from rlberry.agents.bandits import RecursiveIndexAgent
from rlberry.manager import AgentManager, plot_writer_data
import matplotlib.pyplot as plt
from rlberry.wrappers import WriterWrapper


# Agents definition


class UCBAgent(RecursiveIndexAgent):
    """UCB agent for B-subgaussian bandits"""

    name = "UCB Agent"

    def __init__(self, env, B=1, **kwargs):
        def stat_function(stat, Na, action, reward):
            # The statistic is the empirical mean. We compute it recursively.
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


# Parameters of the problem
means = [0, 0.9, 1]  # means of the arms
T = 3000  # Horizon
M = 20  # number of MC simu

# Construction of the experiment

env_ctor = NormalBandit
env_kwargs = {"means": means, "stds": 2 * np.ones(len(means))}

agent = AgentManager(
    UCBAgent,
    (env_ctor, env_kwargs),
    fit_budget=T,
    init_kwargs={"B": 2},
    n_fit=M,
    parallelization="process",
    mp_context="fork",
)
# these parameters should give parallel computing even in notebooks


# Agent training

agent.fit()


# Compute and plot regret
def compute_regret(regret):
    return np.cumsum(np.max(means) - regret)


fig = plt.figure(1, figsize=(5, 3))
ax = plt.gca()
output = plot_writer_data(
    [agent],
    tag="reward",
    preprocess_func=compute_regret,
    title="Cumulative Regret",
    ax=ax,
)
