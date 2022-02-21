"""
=============================================================
Comparison subplots of various index based bandits algorithms
=============================================================

This script Compare several bandits agents and as a sub-product also shows
how to use subplots in with `plot_writer_data`
"""
import numpy as np
from rlberry.envs.bandits import BernoulliBandit
from rlberry.agents.bandits import IndexAgent, RandomizedAgent
from rlberry.manager import AgentManager, plot_writer_data
import matplotlib.pyplot as plt
from rlberry.wrappers import WriterWrapper

# Agents definition
# sphinx_gallery_thumbnail_number = 2


# Parameters of the problem
means = np.array([0.6, 0.6, 0.6, 0.9])  # means of the arms
A = len(means)
T = 2000  # Horizon
M = 10  # number of MC simu

# Construction of the experiment

env_ctor = BernoulliBandit
env_kwargs = {"p": means}


class UCBAgent(IndexAgent):
    name = "UCB"

    def __init__(self, env, **kwargs):
        def index(r, t):
            return np.mean(r) + np.sqrt(np.log(t**2) / (2 * len(r)))

        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="action_and_reward"
        )


class ETCAgent(IndexAgent):
    name = "ETC"

    def __init__(self, env, m=20, **kwargs):
        def index(r, t):
            if t < m * A:
                index = -len(r)  # select an action pulled the least
            else:
                index = np.mean(r, axis=0)
            return index

        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="action_and_reward"
        )


class MOSSAgent(IndexAgent):
    name = "MOSS"

    def __init__(self, env, **kwargs):
        def index(r, t):
            Na = len(r)
            return np.mean(r) + np.sqrt(A / Na * max(0, np.log(T / (A * Na))))

        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="action_and_reward"
        )


class EXP3Agent(RandomizedAgent):
    name = "EXP3"

    def __init__(self, env, **kwargs):
        def index(r, p, t):
            return np.sum(1 - (1 - r) / p)

        def prob(indices, t):
            eta = np.minimum(np.sqrt(self.n_arms * np.log(self.n_arms) / (t + 1)), 1.0)
            w = np.exp(eta * indices)
            w /= w.sum()
            return (1 - eta) * w + eta * np.ones(self.n_arms) / self.n_arms

        RandomizedAgent.__init__(self, env, index, prob, **kwargs)
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="action_and_reward"
        )


Agents_class = [UCBAgent, ETCAgent, MOSSAgent, EXP3Agent]

agents = [
    AgentManager(
        Agent,
        (env_ctor, env_kwargs),
        fit_budget=T,
        n_fit=M,
        parallelization="process",
        mp_context="fork",
    )
    for Agent in Agents_class
]

# these parameters should give parallel computing even in notebooks


# Agent training
for agent in agents:
    agent.fit()


# Compute and plot regret
def compute_regret(rewards):
    return np.cumsum(np.max(means) - rewards)


# Compute and plot (pseudo-)regret
def compute_pseudo_regret(actions):
    return np.cumsum(np.max(means) - means[actions.astype(int)])


output = plot_writer_data(
    agents,
    tag="action",
    preprocess_func=compute_pseudo_regret,
    title="Cumulative Pseudo-Regret",
)

output = plot_writer_data(
    agents,
    tag="reward",
    preprocess_func=compute_regret,
    title="Cumulative Regret",
)


# Compute and plot number of times each arm was selected
def compute_na(actions, a):
    return np.cumsum(actions == a)


fig, axes = plt.subplots(2, 2, sharey=True, figsize=(6, 6))
axes = axes.ravel()
for arm in range(A):
    output = plot_writer_data(
        agents,
        tag="action",
        preprocess_func=lambda actions: compute_na(actions, arm),
        title="Na for arm " + str(arm) + ", mean=" + str(means[arm]),
        ax=axes[arm],
        show=False,
    )
fig.tight_layout()
plt.show()
