"""
=========================
Compare Bandit Algorithms
=========================

This example illustrate the use of compare_agents, a function that uses multiple-testing to assess whether traine agents are
statistically different or not.

Remark that in the case where two agents are not deemed statistically different it can mean either that they are as efficient,
or it can mean that there have not been enough fits to assess the variability of the agents.

"""


import numpy as np

from rlberry.manager.comparison import compare_agents
from rlberry.manager import AgentManager
from rlberry.envs.bandits import BernoulliBandit
from rlberry.wrappers import WriterWrapper
from rlberry.agents.bandits import (
    IndexAgent,
    makeBoundedIMEDIndex,
    makeBoundedMOSSIndex,
    makeBoundedNPTSIndex,
    makeBoundedUCBIndex,
    makeETCIndex,
)

# Parameters of the problem
means = np.array([0.6, 0.6, 0.6, 0.9])  # means of the arms
A = len(means)
T = 3000  # Horizon
N = 50  # number of fits

# Construction of the experiment

env_ctor = BernoulliBandit
env_kwargs = {"p": means}


class UCBAgent(IndexAgent):
    name = "UCB"

    def __init__(self, env, **kwargs):
        index, _ = makeBoundedUCBIndex()
        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


class ETCAgent(IndexAgent):
    name = "ETC"

    def __init__(self, env, m=20, **kwargs):
        index, _ = makeETCIndex(A, m)
        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="action_and_reward"
        )


class MOSSAgent(IndexAgent):
    name = "MOSS"

    def __init__(self, env, **kwargs):
        index, _ = makeBoundedMOSSIndex(T, A)
        IndexAgent.__init__(self, env, index, **kwargs)
        self.env = WriterWrapper(
            self.env, self.writer, write_scalar="action_and_reward"
        )


class IMEDAgent(IndexAgent):
    name = "IMED"

    def __init__(self, env, **kwargs):
        index, tracker_params = makeBoundedIMEDIndex()
        IndexAgent.__init__(self, env, index, tracker_params=tracker_params, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


class NPTSAgent(IndexAgent):
    name = "NPTS"

    def __init__(self, env, **kwargs):
        index, tracker_params = makeBoundedNPTSIndex()
        IndexAgent.__init__(self, env, index, tracker_params=tracker_params, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


Agents_class = [MOSSAgent, IMEDAgent, NPTSAgent, UCBAgent, ETCAgent]

managers = [
    AgentManager(
        Agent,
        train_env=(env_ctor, env_kwargs),
        fit_budget=T,
        parallelization="process",
        mp_context="fork",
        n_fit=N,
    )
    for Agent in Agents_class
]


for manager in managers:
    manager.fit()


def eval_function(manager, eval_budget=None, agent_id=0):
    df = manager.get_writer_data()[agent_id]
    return T * np.max(means) - np.sum(df.loc[df["tag"] == "reward", "value"])


compare_agents(managers, method="permutation", eval_function=eval_function, B=10_000)
