import pytest
from rlberry.agents import *
from rlberry.agents.torch import *
from rlberry.utils import check_rl_agent
from rlberry.agents.features import FeatureMap
import numpy as np


class OneHotFeatureMap(FeatureMap):
    def __init__(self, S, A):
        self.S = S
        self.A = A
        self.shape = (S * A,)

    def map(self, observation, action):
        feat = np.zeros((self.S, self.A))
        feat[observation, action] = 1.0
        return feat.flatten()


class OneHotLSVI(LSVIUCBAgent):
    def __init__(self, env, **kwargs):
        def feature_map_fn(_env):
            return OneHotFeatureMap(5, 2)  # values for Chain

        LSVIUCBAgent.__init__(
            self, env, feature_map_fn=feature_map_fn, horizon=10, **kwargs
        )


FINITE_MDP_AGENTS = [
    ValueIterationAgent,
    MBQVIAgent,
    UCBVIAgent,
    OptQLAgent,
    OneHotLSVI,
    PSRLAgent,
    RLSVIAgent,
]


CONTINUOUS_STATE_AGENTS = [
    RSUCBVIAgent,
    RSKernelUCBVIAgent,
    # DQNAgent,  # For now, DQN does not work. Problem with reproducibility.
    PPOAgent,
    AVECPPOAgent,
    REINFORCEAgent,
]


@pytest.mark.parametrize("Agent", FINITE_MDP_AGENTS)
def test_finite_state_agent(Agent):
    check_rl_agent(Agent, continuous_state=False)


@pytest.mark.parametrize("Agent", CONTINUOUS_STATE_AGENTS)
def test_continuous_state_agent(Agent):
    check_rl_agent(Agent, continuous_state=True)
