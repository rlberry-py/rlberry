from rlberry.envs.bandits import NormalBandit, BernoulliBandit
from rlberry.agents.bandits import (
    IndexAgent,
    RandomizedAgent,
    TSAgent,
    BanditWithSimplePolicy,
    makeBoundedMOSSIndex,
    makeBoundedUCBIndex,
    makeETCIndex,
    makeEXP3Index,
    makeSubgaussianMOSSIndex,
    makeSubgaussianUCBIndex,
)
from rlberry.utils import check_bandit_agent


TEST_SEED = 42


def test_base_bandit():
    assert check_bandit_agent(BanditWithSimplePolicy, NormalBandit, seed=TEST_SEED)


bounded_indices = [makeBoundedUCBIndex, makeBoundedMOSSIndex]
subgaussian_indices = [makeSubgaussianUCBIndex, makeSubgaussianMOSSIndex]
misc_indices = [makeETCIndex]


def test_bounded_indices():
    for makeIndex in bounded_indices:

        class Agent(IndexAgent):
            def __init__(self, env, B=1, **kwargs):
                index = makeIndex()
                IndexAgent.__init__(self, env, index, **kwargs)

        assert check_bandit_agent(Agent, BernoulliBandit, seed=TEST_SEED)


def test_subgaussian_indices():
    for makeIndex in subgaussian_indices:

        class Agent(IndexAgent):
            def __init__(self, env, B=1, **kwargs):
                index = makeIndex()
                IndexAgent.__init__(self, env, index, **kwargs)

        assert check_bandit_agent(Agent, NormalBandit, seed=TEST_SEED)


def test_misc_indices():
    for makeIndex in misc_indices:

        class Agent(IndexAgent):
            def __init__(self, env, B=1, **kwargs):
                index = makeIndex()
                IndexAgent.__init__(self, env, index, **kwargs)

        assert check_bandit_agent(Agent, BernoulliBandit, seed=TEST_SEED)


def test_randomized_bandits():
    class EXP3Agent(RandomizedAgent):
        name = "EXP3"

        def __init__(self, env, **kwargs):
            prob = makeEXP3Index()
            RandomizedAgent.__init__(self, env, prob, **kwargs)

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
