import numpy as np
from rlberry.envs.bandits import NormalBandit, BernoulliBandit
from rlberry.agents.bandits import (
    IndexAgent,
    RandomizedAgent,
    RecursiveIndexAgent,
    TSAgent,
    BanditWithSimplePolicy,
    makeBoundedUCBIndex,
)
from rlberry.manager import AgentManager
from rlberry.utils import check_bandit_agent


TEST_SEED = 42


class UCBAgent(IndexAgent):
    name = "UCB Agent"

    def __init__(self, env, B=1, **kwargs):
        index = makeBoundedUCBIndex()
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

        index = makeBoundedUCBIndex()

        RecursiveIndexAgent.__init__(self, env, stat_function, index, **kwargs)


def test_base_bandit():
    assert check_bandit_agent(BanditWithSimplePolicy, NormalBandit, seed=TEST_SEED)


def test_index_bandits():
    assert check_bandit_agent(UCBAgent, NormalBandit, seed=TEST_SEED)
    assert check_bandit_agent(RecursiveUCBAgent, NormalBandit, seed=TEST_SEED)


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


def test_randomized_bandits():
    assert check_bandit_agent(EXP3Agent, BernoulliBandit, seed=TEST_SEED)


def test_TS():
    class TSAgent_normal(TSAgent):
        name = "TSAgent"

        def __init__(self, env, **kwargs):
            TSAgent.__init__(self, env, "gaussian", **kwargs)

    assert check_bandit_agent(TSAgent_normal, NormalBandit, seed=TEST_SEED)

    class TSAgent_beta(TSAgent):
        name = "TSAgent"

        def __init__(self, env, **kwargs):
            TSAgent.__init__(self, env, "beta", **kwargs)

    assert check_bandit_agent(TSAgent_beta, BernoulliBandit, TEST_SEED)


def test_recursive_vs_not_recursive():
    env_ctor = NormalBandit
    env_kwargs = {}

    agent1 = AgentManager(
        UCBAgent, (env_ctor, env_kwargs), fit_budget=10, n_fit=1, seed=TEST_SEED
    )

    agent2 = AgentManager(
        RecursiveUCBAgent,
        (env_ctor, env_kwargs),
        fit_budget=10,
        n_fit=1,
        seed=TEST_SEED,
    )

    agent1.fit()
    agent2.fit()
    env = env_ctor(**env_kwargs)
    state = env.reset()
    for _ in range(5):
        # test reproducibility on 5 actions
        action1 = agent1.agent_handlers[0].policy(state)
        action2 = agent2.agent_handlers[0].policy(state)
        assert action1 == action2
