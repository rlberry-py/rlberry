"""
===============================================
Tests some agent and env from rlberry only (no rlberry-scool or rlberry research)
===============================================

"""

from rlberry.utils.check_env import check_env, check_gym_env
from rlberry.utils.check_agent import check_rl_agent
from rlberry.envs import gym_make
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import A2C

# from stable_baselines3 import DQN
import pytest


FROZEN_LAKE_CONSTR = (
    gym_make,
    dict(id="FrozenLake-v1", wrap_spaces=True, is_slippery=False),
)
CART_POLE_CONSTR = (gym_make, dict(id="CartPole-v1", wrap_spaces=True))

ALL_ENVS = [
    FROZEN_LAKE_CONSTR,
    CART_POLE_CONSTR,
]


@pytest.mark.parametrize("Env", ALL_ENVS)
def test_env(Env):
    current_env = Env[0](**Env[1])
    check_env(current_env)
    # check_rlberry_env(current_env)
    check_gym_env(current_env)


A2C_INIT_KWARGS = {"algo_cls": A2C, "policy": "MlpPolicy", "verbose": 1}
# DQN_INIT_KWARGS = {"algo_cls": DQN, "policy": "MlpPolicy", "verbose": 1}

AGENTS_WITH_ENV = [
    (A2C_INIT_KWARGS, CART_POLE_CONSTR),
    # (DQN_INIT_KWARGS,FROZEN_LAKE_CONSTR),
]


@pytest.mark.parametrize("agent_kwargs,env", AGENTS_WITH_ENV)
def test_rlberry_agent(agent_kwargs, env):
    check_rl_agent(
        StableBaselinesAgent,
        env=env,
        init_kwargs=agent_kwargs,
    )
