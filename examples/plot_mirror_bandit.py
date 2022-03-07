"""
========================================================
A demo of Bandit BAI on a real dataset to select mirrors
========================================================
In this exemple we use a sequential halving agent to find the best server
to download ubuntu from amlong a choice of 8 french servers.
"""

from rlberry.envs.bandits import MirrorBandit
from rlberry.agents.bandits import SeqHalvAgent
from rlberry.manager import AgentManager, read_writer_data
import numpy as np

env_ctor = MirrorBandit
env_kwargs = {}

agent = AgentManager(
    SeqHalvAgent,
    (env_ctor, env_kwargs),
    fit_budget=200,
    init_kwargs={"mean_est_function": lambda x: np.median(x)},
    n_fit=1,
    agent_name="SH",
)
agent.fit()

rewards = read_writer_data([agent], tag="reward")["value"]
actions = read_writer_data([agent], tag="action")["value"]

import matplotlib.pyplot as plt

plt.boxplot([-rewards[actions == a] for a in range(6)])
plt.show()

print("optimal action is ", agent.agent_handlers[0].optimal_action + 1)
