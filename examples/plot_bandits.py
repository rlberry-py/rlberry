"""
=============================
UCB Bandit cumulative regret
=============================

This script shows how to define a
"""

#! pip install git+https://github.com/TimotheeMathieu/rlberry@bandits-perso
import numpy as np
from rlberry.envs.bandits import NormalBandit
from rlberry.agents.bandits import IndexAgent, RecursiveIndexAgent
from rlberry.manager import AgentManager, plot_writer_data
import matplotlib.pyplot as plt
from rlberry.wrappers import WriterWrapper


############ Agents ###############################


class UCBAgent(IndexAgent):
    """Basic UCB agent for sub-gaussian rv."""

    name = "UCB Agent"

    def __init__(self, env, B=1, **kwargs):
        # upper bound mean + B*0.5*sqrt(2*log(t**2)/n)
        # if bounded by B or B subgaussian
        index = lambda r, t: np.mean(r) + B * 0.5 * np.sqrt(2 * np.log(t ** 2) / len(r))
        IndexAgent.__init__(self, env, index, **kwargs)
        # record rewards
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


class RecursiveUCBAgent(RecursiveIndexAgent):
    """Same as above but defined recursively"""

    name = "Recursive UCB Agent"

    def __init__(self, env, B=1, **kwargs):
        # upper bound mean + B*0.5*sqrt(2*log(t**2)/n)
        # if bounded by B or B subgaussian
        def stat_function(stat, Na, action, reward):
            if stat is None:
                stat = np.zeros(len(Na))
            # update the empirical mean
            stat[action] = (Na[action] - 1) / Na[action] * stat[action] + reward
            return stat

        index = lambda stat, Na, t: stat + B * 0.5 * np.sqrt(2 * np.log(t ** 2) / Na)
        RecursiveIndexAgent.__init__(self, env, stat_function, index, **kwargs)
        # record rewards
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


class NaiveAgent(IndexAgent):
    name = "Naive Agent"

    def __init__(self, env, **kwargs):
        # choose the arm with the largest empirical mean
        IndexAgent.__init__(self, env, lambda r, t: np.mean(r), **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


############# Parameters ###########################
means = [0, 0.9, 1]
T = 3000  # Horizon
M = 20  # number of MC simu


############# Construction of the experiment ########


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


################ Agent training ################

agent1.fit()
agent1bis.fit()
agent2.fit()

################# Compute and plot ###################
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
# you can customize the axis ax here to customize the plot
# or you can use output, which is a pandat DataFrame to do the plot yourself.
