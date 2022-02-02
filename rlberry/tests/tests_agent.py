import pytest
from rlberry.agents import *
from rlberry.agents.torch import *
from rlberry.utils.check_discrete_action_agent import (
    check_finiteMDP_agent,
    check_continuous_state_agent,
)

FINITE_MDP_AGENTS = [
    ValueIterationAgent,
    MBQVIAgent,
    UCBVIAgent,
    OptQLAgent,
    # LSVIUCBAgent, # for now excluded because feature map not specified
]


CONTINUOUS_STATE_AGENTS = [
    RSUCBVIAgent,
    RSKernelUCBVIAgent,
    DQNAgent,
    REINFORCEAgent,
]


@pytest.mark.parametrize("Agent", FINITE_MDP_AGENTS)
def test_finite_state_agent(Agent):
    assert check_finiteMDP_agent(Agent)


@pytest.mark.parametrize("Agent", CONTINUOUS_STATE_AGENTS)
def test_continuous_state_agent(Agent):
    assert check_continuous_state_agent(Agent)
