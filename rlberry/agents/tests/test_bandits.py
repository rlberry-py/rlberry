from rlberry.envs.bandits import NormalBandit, BernoulliBandit
from rlberry.agents.bandits import (
    IndexAgent,
    RandomizedAgent,
    TSAgent,
    BanditWithSimplePolicy,
    makeBoundedIMEDIndex,
    makeBoundedMOSSIndex,
    makeBoundedNPTSIndex,
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


bounded_indices = {
    "IMED": makeBoundedIMEDIndex,
    "MOSS": makeBoundedMOSSIndex,
    "NPTS": makeBoundedNPTSIndex,
    "UCB": makeBoundedUCBIndex,
}
subgaussian_indices = {
    "UCB": makeSubgaussianUCBIndex,
    "MOSS": makeSubgaussianMOSSIndex,
}
misc_indices = {
    "ETC": makeETCIndex,
}


def test_bounded_indices():
    for agent_name, makeIndex in bounded_indices.items():

        class Agent(IndexAgent):
            name = agent_name

            def __init__(self, env, B=1, **kwargs):
                index, tracker_params = makeIndex()
                IndexAgent.__init__(
                    self, env, index, tracker_params=tracker_params, **kwargs
                )

        assert check_bandit_agent(Agent, BernoulliBandit, seed=TEST_SEED)


def test_subgaussian_indices():
    for agent_name, makeIndex in subgaussian_indices.items():

        class Agent(IndexAgent):
            name = agent_name

            def __init__(self, env, B=1, **kwargs):
                index, tracker_params = makeIndex()
                IndexAgent.__init__(
                    self, env, index, tracker_params=tracker_params, **kwargs
                )

        assert check_bandit_agent(Agent, NormalBandit, seed=TEST_SEED)


def test_misc_indices():
    for agent_name, makeIndex in misc_indices.items():

        class Agent(IndexAgent):
            name = agent_name

            def __init__(self, env, B=1, **kwargs):
                index, tracker_params = makeIndex()
                IndexAgent.__init__(
                    self, env, index, tracker_params=tracker_params, **kwargs
                )

        assert check_bandit_agent(Agent, BernoulliBandit, seed=TEST_SEED)


def test_randomized_bandits():
    class EXP3Agent(RandomizedAgent):
        name = "EXP3"

        def __init__(self, env, **kwargs):
            prob, tracker_params = makeEXP3Index()
            RandomizedAgent.__init__(
                self, env, prob, tracker_params=tracker_params, **kwargs
            )

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
