import numpy as np
from rlberry.envs.bandits import NormalBandit
from rlberry.agents.bandits import IndexAgent, RecursiveIndexAgent
from rlberry.manager import AgentManager
from rlberry.utils import check_bandit_agent


class UCBAgent(IndexAgent):
    name = "UCB Agent"

    def __init__(self, env, B=1, **kwargs):
        index = lambda r, t: np.mean(r) + B * np.sqrt(2 * np.log(t**2) / len(r))
        IndexAgent.__init__(self, env, index, **kwargs)


class RecursiveUCBAgent(RecursiveIndexAgent):
    name = "Recursive UCB Agent"

    def __init__(self, env, B=1, **kwargs):
        def stat_function(stat, Na, action, reward):
            if stat is None:
                stat = np.zeros(len(Na))
            stat[action] = (Na[action] - 1) / Na[action] * stat[action] + reward / Na[
                action
            ]
            return stat

        index = lambda stat, Na, t: stat + B * np.sqrt(2 * np.log(t**2) / Na)
        RecursiveIndexAgent.__init__(self, env, stat_function, index, **kwargs)


def test_index_bandits():
    assert check_bandit_agent(UCBAgent)
    assert check_bandit_agent(RecursiveUCBAgent)


def test_recursive_vs_not_recursive():
    env_ctor = NormalBandit
    env_kwargs = {}

    agent1 = AgentManager(
        UCBAgent, (env_ctor, env_kwargs), fit_budget=10, n_fit=1, seed=42
    )
    agent2 = AgentManager(
        RecursiveUCBAgent, (env_ctor, env_kwargs), fit_budget=10, n_fit=1, seed=42
    )

    agent1.fit()
    agent2.fit()
    env = env_ctor(**env_kwargs)
    state = env.reset()
    result = True
    for f in range(5):
        # test reproducibility on 5 actions
        action1 = agent1.agent_handlers[0].policy(state)
        action2 = agent2.agent_handlers[0].policy(state)
        assert action1 == action2
