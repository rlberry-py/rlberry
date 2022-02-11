"""
=============================================================
Comparison subplots of various index based bandits algorithms
=============================================================

This script Compare several bandits agents and as a sub-product also shows
how to use subplots in with `plot_writer_data`
"""
import numpy as np
from rlberry.envs.bandits import BernoulliBandit
from rlberry.agents.bandits import IndexAgent
from rlberry.manager import AgentManager, plot_writer_data
import matplotlib.pyplot as plt
from rlberry.wrappers import WriterWrapper

# Agents definition
# sphinx_gallery_thumbnail_number = 2


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

    def __init__(self, env, m=10, **kwargs):
        def index(r, t):
            A = 4
            indexes = np.zeros(A)
            if t < m * A:
                indexes[(t % A)] = 1
            else:
                indexes = np.mean(r, axis=0)

        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="action_and_reward"
        )


class MOSSAgent(IndexAgent):
    name = "MOSS"

    def __init__(self, env, **kwargs):
        def index(r, t, n, A):
            Na = len(r)
            return np.mean(r) + np.sqrt(4 / Na * max(0, np.log(n / (A * Na))))

        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="action_and_reward"
        )

    def fit(self, budget=None, **kwargs):
        horizon = budget
        rewards = np.zeros(horizon)
        actions = np.ones(horizon) * np.nan

        indexes = np.inf * np.ones(self.n_arms)
        for ep in range(horizon):
            if self.total_time < self.n_arms:
                action = self.total_time
            else:
                indexes = self.get_indexes(rewards, actions, ep, horizon)
                action = np.argmax(indexes)
            self.total_time += 1
            _, reward, _, _ = self.env.step(action)
            rewards[ep] = reward
            actions[ep] = action

        self.optimal_action = np.argmax(indexes)
        info = {"episode_reward": np.sum(rewards)}
        return info

    def get_indexes(self, rewards, actions, ep, n):
        indexes = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            indexes[a] = self.index_function(rewards[actions == a], ep, n, self.n_arms)
        return indexes


# Parameters of the problem
means = [0.8, 0.8, 0.8, 1]  # means of the arms
T = 2000  # Horizon
M = 5  # number of MC simu

# Construction of the experiment

env_ctor = BernoulliBandit
env_kwargs = {"p": means}

agent1 = AgentManager(
    UCBAgent,
    (env_ctor, env_kwargs),
    fit_budget=T,
    n_fit=M,
    parallelization="process",
    mp_context="fork",
)
agent2 = AgentManager(
    ETCAgent,
    (env_ctor, env_kwargs),
    fit_budget=T,
    n_fit=M,
    parallelization="process",
    mp_context="fork",
)
agent3 = AgentManager(
    MOSSAgent,
    (env_ctor, env_kwargs),
    fit_budget=T,
    n_fit=M,
    parallelization="process",
    mp_context="fork",
)
agents = [agent1, agent2, agent3]
# these parameters should give parallel computing even in notebooks


# Agent training
for agent in agents:
    agent.fit()


# Compute and plot regret
def compute_regret(regret):
    return np.cumsum(np.max(means) - regret)


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
for arm in range(4):
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
