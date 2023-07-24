"""
===============================================
Tests for the installation without extra (StableBaselines3, torch, optuna, ...)
===============================================
tests based on test_agent.py and test_envs.py

"""


import sys

import numpy as np
import pytest

import rlberry.agents as agents
from rlberry.agents.features import FeatureMap
from rlberry.utils.check_agent import check_rl_agent, check_rlberry_agent


class OneHotFeatureMap(FeatureMap):
    def __init__(self, S, A):
        self.S = S
        self.A = A
        self.shape = (S * A,)

    def map(self, observation, action):
        feat = np.zeros((self.S, self.A))
        feat[observation, action] = 1.0
        return feat.flatten()


# LSVIUCBAgent needs a feature map function to work.
class OneHotLSVI(agents.LSVIUCBAgent):
    def __init__(self, env, **kwargs):
        def feature_map_fn(_env):
            return OneHotFeatureMap(5, 2)  # values for Chain

        agents.LSVIUCBAgent.__init__(
            self, env, feature_map_fn=feature_map_fn, horizon=10, **kwargs
        )


FINITE_MDP_AGENTS = [
    agents.QLAgent,
    agents.SARSAAgent,
    agents.ValueIterationAgent,
    agents.MBQVIAgent,
    agents.UCBVIAgent,
    agents.OptQLAgent,
    agents.PSRLAgent,
    agents.RLSVIAgent,
    OneHotLSVI,
]


CONTINUOUS_STATE_AGENTS = [
    agents.RSUCBVIAgent,
    agents.RSKernelUCBVIAgent,
]


@pytest.mark.parametrize("agent", FINITE_MDP_AGENTS)
def test_finite_state_agent(agent):
    check_rl_agent(agent, env="discrete_state")
    check_rlberry_agent(agent, env="discrete_state")


@pytest.mark.xfail(sys.platform == "win32", reason="bug with windows???")
@pytest.mark.parametrize("agent", CONTINUOUS_STATE_AGENTS)
def test_continuous_state_agent(agent):
    check_rl_agent(agent, env="continuous_state")
    check_rlberry_agent(agent, env="continuous_state")
