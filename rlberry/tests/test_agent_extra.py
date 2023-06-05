import pytest
import rlberry.agents as agents
import rlberry.agents.torch as torch_agents
from rlberry.utils.check_agent import (
    check_rl_agent,
    check_rlberry_agent,
    check_vectorized_env_agent,
    check_hyperparam_optimisation_agent,
)
from rlberry.agents.features import FeatureMap
import numpy as np
import sys


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


# No agent "FINITE_MDP" in extra
# FINITE_MDP_AGENTS = [
# ]


CONTINUOUS_STATE_AGENTS = [
    torch_agents.DQNAgent,
    torch_agents.MunchausenDQNAgent,
    torch_agents.REINFORCEAgent,
    torch_agents.PPOAgent,
    torch_agents.A2CAgent,
]


HYPERPARAM_OPTI_AGENTS = [
    torch_agents.PPOAgent,
    torch_agents.REINFORCEAgent,
    torch_agents.A2CAgent,
]


MULTI_ENV_AGENTS = [
    torch_agents.PPOAgent,
]

# No agent "FINITE_MDP" in extra
# @pytest.mark.parametrize("agent", FINITE_MDP_AGENTS)
# def test_finite_state_agent(agent):
#     check_rl_agent(agent, env="discrete_state")
#     check_rlberry_agent(agent, env="discrete_state")


@pytest.mark.xfail(sys.platform == "win32", reason="bug with windows???")
@pytest.mark.parametrize("agent", CONTINUOUS_STATE_AGENTS)
def test_continuous_state_agent(agent):
    check_rl_agent(agent, env="continuous_state")
    check_rlberry_agent(agent, env="continuous_state")


@pytest.mark.xfail(sys.platform == "win32", reason="bug with windows???")
@pytest.mark.parametrize("agent", MULTI_ENV_AGENTS)
def test_continuous_vectorized_env_agent(agent):
    check_vectorized_env_agent(agent, env="vectorized_env_continuous")


@pytest.mark.xfail(sys.platform == "win32", reason="bug with windows???")
@pytest.mark.parametrize("agent", HYPERPARAM_OPTI_AGENTS)
def test_hyperparam_optimisation_agent(agent):
    check_hyperparam_optimisation_agent(agent, env="continuous_state")
