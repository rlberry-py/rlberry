"""
===============================================
Tests for the installation without extra (StableBaselines3, torch, optuna, ...)
===============================================
tests based on test_agent.py and test_envs.py

"""


import pytest
import numpy as np
import sys

import rlberry.agents as agents
from rlberry.agents.features import FeatureMap

from rlberry.envs import Acrobot
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.envs.classic_control import MountainCar, SpringCartPole
from rlberry.envs.finite import Chain, GridWorld

from rlberry.utils.check_agent import (
    check_rl_agent,
    check_rlberry_agent,
)
from rlberry.utils.check_env import check_env, check_rlberry_env


#### TESTS AGENTS ####

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



#### TESTS ENVS ####


ALL_ENVS = [
    Acrobot,
    PBall2D,
    TwinRooms,
    AppleGold,
    NRoom,
    MountainCar,
    Chain,
    GridWorld,
    SpringCartPole,
]


@pytest.mark.parametrize("Env", ALL_ENVS)
def test_env(Env):
    check_env(Env())
    check_rlberry_env(Env())


