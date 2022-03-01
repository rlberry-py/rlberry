from rlberry.envs.bandits import NormalBandit, BernoulliBandit
from rlberry.agents.bandits import (
    IndexAgent,
    makeETCIndex,
    makeBoundedUCBIndex,
    makeBoundedMOSSIndex,
    makeSubgaussianUCBIndex,
    makeSubgaussianMOSSIndex,
)
from rlberry.utils import check_bandit_agent


TEST_SEED = 42

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
