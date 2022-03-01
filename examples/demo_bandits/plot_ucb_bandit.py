"""
=============================
UCB Bandit cumulative regret
=============================

This script shows how to define a bandit environment and an UCB Index-based algorithm.
"""

import numpy as np
from rlberry.envs.bandits import NormalBandit
from rlberry.agents.bandits import IndexAgent
from rlberry.manager import AgentManager, plot_writer_data
import matplotlib.pyplot as plt
from rlberry.wrappers import WriterWrapper


# Agents definition


class UCBAgent(IndexAgent):
    """UCB agent for B-subgaussian bandits"""

    name = "UCB Agent"

    def __init__(self, env, B=1, **kwargs):
        def index(tr):
            return tr.mu_hats + np.sqrt(np.log(tr.t**2) / (2 * tr.n_pulls))

        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="action")


# Parameters of the problem
means = np.array([0, 0.9, 1])  # means of the arms
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


# Compute and plot (pseudo-)regret
def compute_pseudo_regret(actions):
    return np.cumsum(np.max(means) - means[actions.astype(int)])


fig = plt.figure(1, figsize=(5, 3))
ax = plt.gca()
output = plot_writer_data(
    [agent],
    tag="action",
    preprocess_func=compute_pseudo_regret,
    title="Cumulative Pseudo-Regret",
    ax=ax,
)
