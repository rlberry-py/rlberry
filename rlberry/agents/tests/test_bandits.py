from rlberry.agents.bandits import (
    BanditWithSimplePolicy,
    IndexAgent,
    RandomizedAgent,
    TSAgent,
    makeBetaPrior,
    makeBoundedIMEDIndex,
    makeBoundedMOSSIndex,
    makeBoundedNPTSIndex,
    makeBoundedUCBIndex,
    makeBoundedUCBVIndex,
    makeETCIndex,
    makeEXP3Index,
    makeGaussianPrior,
    makeSubgaussianMOSSIndex,
    makeSubgaussianUCBIndex,
)
from rlberry.envs.bandits import BernoulliBandit, NormalBandit
from rlberry.utils import check_bandit_agent

TEST_SEED = 42


def test_base_bandit():
    assert check_bandit_agent(BanditWithSimplePolicy, NormalBandit, seed=TEST_SEED)


bounded_indices = {
    "IMED": makeBoundedIMEDIndex,
    "MOSS": makeBoundedMOSSIndex,
    "NPTS": makeBoundedNPTSIndex,
    "UCB": makeBoundedUCBIndex,
    "UCBV": makeBoundedUCBVIndex,
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

            def __init__(self, env, **kwargs):
                index, tracker_params = makeIndex()
                IndexAgent.__init__(
                    self, env, index, tracker_params=tracker_params, **kwargs
                )

        assert check_bandit_agent(
            Agent, BernoulliBandit, seed=TEST_SEED
        ), "Agent not reproducible"


def test_subgaussian_indices():
    for agent_name, makeIndex in subgaussian_indices.items():

        class Agent(IndexAgent):
            name = agent_name

            def __init__(self, env, **kwargs):
                index, tracker_params = makeIndex()
                IndexAgent.__init__(
                    self, env, index, tracker_params=tracker_params, **kwargs
                )

        assert check_bandit_agent(
            Agent, NormalBandit, seed=TEST_SEED
        ), "Agent not reproducible"


def test_misc_indices():
    for agent_name, makeIndex in misc_indices.items():

        class Agent(IndexAgent):
            name = agent_name

            def __init__(self, env, **kwargs):
                index, tracker_params = makeIndex()
                IndexAgent.__init__(
                    self, env, index, tracker_params=tracker_params, **kwargs
                )

        assert check_bandit_agent(
            Agent, BernoulliBandit, seed=TEST_SEED
        ), "Agent not reproducible"


def test_randomized_bandits():
    class EXP3Agent(RandomizedAgent):
        name = "EXP3"

        def __init__(self, env, **kwargs):
            prob, tracker_params = makeEXP3Index()
            RandomizedAgent.__init__(
                self, env, prob, tracker_params=tracker_params, **kwargs
            )

    assert check_bandit_agent(
        EXP3Agent, BernoulliBandit, seed=TEST_SEED
    ), "Agent not reproducible"


priors = {
    "Beta": (makeBetaPrior, BernoulliBandit),
    "Gaussian": (makeGaussianPrior, NormalBandit),
}


def test_TS():
    for agent_name, (makePrior, Bandit) in priors.items():

        class Agent(TSAgent):
            name = agent_name

            def __init__(self, env, **kwargs):
                prior_info, tracker_params = makePrior()
                TSAgent.__init__(
                    self, env, prior_info, tracker_params=tracker_params, **kwargs
                )

        assert check_bandit_agent(
            Agent, Bandit, seed=TEST_SEED
        ), "Agent not reproducible"
