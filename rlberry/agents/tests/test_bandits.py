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


def test_bandits():
    means = [0, 0.9, 1]

    ############# Construction of the experiment ########

    env_ctor = NormalBandit
    env_kwargs = {"means": means, "stds": 2 * np.ones(len(means))}

    agent1 = AgentManager(
        UCBAgent,
        (env_ctor, env_kwargs),
        fit_budget=10,
        init_kwargs={"B": 2},
        n_fit=2,
        seed=42,
    )

    agent1bis = AgentManager(
        RecursiveUCBAgent,
        (env_ctor, env_kwargs),
        fit_budget=10,
        init_kwargs={"B": 2},
        n_fit=2,
        seed=42,
    )
    agent1.fit()
    agent1bis.fit()
