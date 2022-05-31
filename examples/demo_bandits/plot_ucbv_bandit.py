"""
=================================================
Comparison plot of UCB and UCBV bandit algorithms
=================================================

This script Compare UCB with UCBV algorithm on a Gaussian Bandit
"""
import numpy as np
from rlberry.envs.bandits import NormalBandit
from rlberry.manager import AgentManager, plot_writer_data
from rlberry.wrappers import WriterWrapper
from rlberry.agents.bandits import IndexAgent, makeSubgaussianUCBIndex

# Agents definition
# sphinx_gallery_thumbnail_number = 2


# Parameters of the problem
means = np.array([0.6, 0.6, 0.6, 0.9])  # means of the arms
A = len(means)
T = 2000  # Horizon
M = 10  # number of MC simu

# Construction of the experiment

env_ctor = NormalBandit
env_kwargs = {"means": means}


class UCBAgent(IndexAgent):
    name = "UCB"

    def __init__(self, env, **kwargs):
        index, _ = makeSubgaussianUCBIndex()
        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="action_and_reward"
        )


class UCBVAgent(IndexAgent):
    """
    UCBV algorithm from [1].

     References
    ----------
    .. [1] Audibert, J. Y., Munos, R., & Szepesvári, C. (2009).
        Exploration–exploitation tradeoff using variance estimates
        in multi-armed bandits. Theoretical Computer Science, 410(19), 1876-1902.
    """

    name = "UCBV"

    def __init__(self, env, **kwargs):
        def update_fun(tr, arm):
            """
            Sequentially add variance estimate to tracker
            """
            if tr.n_pulls(arm) == 1:
                tr.add_scalars(arm, {"v_hat": 0})
            else:
                reward = tr.read_last_tag_value("reward", arm)
                old_muhat = (tr.total_reward(arm) - reward) / (
                    tr.n_pulls(arm) - 1
                )  # compute mu at time n-1
                new_muhat = tr.mu_hat(arm)
                old_vhat = tr.read_last_tag_value("v_hat", arm)
                new_vhat = (
                    old_vhat
                    + ((reward - old_muhat) * (reward - new_muhat) - old_vhat) / tr.t
                )
                tr.add_scalars(arm, {"v_hat": new_vhat})

        def index(tr):
            delta = lambda t: 1 / (1 + (t + 1) * np.log(t + 1) ** 2)
            return [
                tr.mu_hat(arm)
                + np.sqrt(
                    2
                    * tr.read_last_tag_value("v_hat", arm)
                    * np.log(1 / delta(tr.t))
                    / tr.n_pulls(arm)
                )
                + 3 * np.log(1 / delta(tr.t)) / tr.n_pulls(arm)
                for arm in tr.arms
            ]

        IndexAgent.__init__(self, env, index, tracker_update=update_fun, **kwargs)
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="action_and_reward"
        )


Agents_class = [UCBAgent, UCBVAgent]

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


output = plot_writer_data(
    agents,
    tag="reward",
    preprocess_func=compute_regret,
    title="Cumulative Regret",
)
