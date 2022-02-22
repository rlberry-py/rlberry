import pytest
import rlberry.agents as agents
import rlberry.agents.torch as torch_agents
from rlberry.utils.check_agent import (
    check_rl_agent,
    check_save_load,
)
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
    torch_agents.DQNAgent,
    torch_agents.PPOAgent,
    torch_agents.AVECPPOAgent,
    torch_agents.REINFORCEAgent,
]


@pytest.mark.parametrize("Agent", FINITE_MDP_AGENTS)
def test_finite_state_agent(Agent):
    check_rl_agent(Agent, env="discrete_state")


@pytest.mark.parametrize("Agent", CONTINUOUS_STATE_AGENTS)
def test_continuous_state_agent(Agent):
    check_rl_agent(Agent, env="continuous_state")


def test_dqn():
    check_save_load(torch_agents.DQNAgent, env="continuous_state")
