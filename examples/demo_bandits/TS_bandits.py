"""
=========================================================================
Comparison of Thompson sampling and UCB on Bernoulli and Gaussian bandits
=========================================================================

This script shows how to use Thompson sampling on two examples: Bernoulli and Gaussian bandits.
"""

import numpy as np
from rlberry.envs.bandits import BernoulliBandit, NormalBandit
from rlberry.agents.bandits import (
    IndexAgent,
    TSAgent,
    makeBoundedUCBIndex,
    makeSubgaussianUCBIndex,
    makeBetaPrior,
    makeGaussianPrior,
)
from rlberry.manager import AgentManager, plot_writer_data
from rlberry.wrappers import WriterWrapper


# Bernoulli

# Agents definition


class BernoulliTSAgent(TSAgent):
    """Thompson sampling for Bernoulli bandit"""

    name = "Bernoulli TS Agent"

    def __init__(self, env, **kwargs):
        prior, _ = makeBetaPrior()
        TSAgent.__init__(self, env, prior, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="action")


class BernoulliUCBAgent(IndexAgent):
    """UCB agent for bounded bandits"""

    name = "Bernoulli UCB Agent"

    def __init__(self, env, **kwargs):
        index, _ = makeBoundedUCBIndex(1, 0)
        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="action")


# Parameters of the problem
means = np.array([0.8, 0.8, 0.9, 1])  # means of the arms
A = len(means)
T = 2000  # Horizon
M = 1  # number of MC simu

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
    for Agent in [BernoulliUCBAgent, BernoulliTSAgent]
]

# Agent training

for agent in agents:
    agent.fit()


# Compute and plot (pseudo-)regret
def compute_pseudo_regret(actions):
    return np.cumsum(np.max(means) - means[actions.astype(int)])


output = plot_writer_data(
    agents,
    tag="action",
    preprocess_func=compute_pseudo_regret,
    title="Cumulative Pseudo-Regret",
    smooth_weight=0.9,
)


# Gaussian


class GaussianTSAgent(TSAgent):
    """Thompson sampling for Gaussian bandit"""

    name = "Gaussian TS Agent"

    def __init__(self, env, sigma=1.0, **kwargs):
        prior, _ = makeGaussianPrior(sigma)
        TSAgent.__init__(self, env, prior, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="action")


class GaussianUCBAgent(IndexAgent):
    """UCB agent for Gaussian bandits"""

    name = "Gaussian UCB Agent"

    def __init__(self, env, sigma=1.0, **kwargs):
        index, _ = makeSubgaussianUCBIndex(sigma)
        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="action")


# Parameters of the problem
means = np.array([0.3, 0.5])  # means of the arms
sigma = 1.0  # means of the arms
A = len(means)
T = 2000  # Horizon
M = 10  # number of MC simu

# Construction of the experiment

env_ctor = NormalBandit
env_kwargs = {"means": means, "stds": sigma * np.ones(A)}

agents = [
    AgentManager(
        Agent,
        (env_ctor, env_kwargs),
        fit_budget=T,
        n_fit=M,
    )
    for Agent in [GaussianUCBAgent, GaussianTSAgent]
]

# Agent training

for agent in agents:
    agent.fit()


# Compute and plot (pseudo-)regret
def compute_pseudo_regret(actions):
    return np.cumsum(np.max(means) - means[actions.astype(int)])


output = plot_writer_data(
    agents,
    tag="action",
    preprocess_func=compute_pseudo_regret,
    title="Cumulative Pseudo-Regret",
)
