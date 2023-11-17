"""
===============================================
Tests some agent and env from rlberry only (no rlberry-scool or rlberry research)
===============================================

"""

from rlberry.utils.check_env import check_env, check_gym_env
from rlberry.utils.check_agent import check_rl_agent
from rlberry.envs import gym_make, atari_make
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import A2C

from stable_baselines3 import DQN
import pytest


import gymnasium as gym
import numpy as np
from typing import Tuple


class CustomDummyEnv(gym.Env):
    def __init__(self):
        obs_dict = dict(
            board=gym.spaces.Box(low=0, high=1, shape=(8 * 8,), dtype=bool),
            player=gym.spaces.Discrete(8),
        )
        self.observation_space = gym.spaces.Dict(obs_dict)
        self.action_space = gym.spaces.MultiDiscrete([8, 8])

    def reset(self):
        return self._obs(), {}

    def _obs(self):
        return {"board": np.zeros(shape=(8, 8), dtype=bool).flatten(), "player": 1}

    def step(self, action: Tuple[int, int]):
        reward = 0.2
        terminated = False
        truncated = False
        info = {}
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        print("hi")

    def reseed(self, seed):
        print("reseed")


FROZEN_LAKE_CONSTR = (
    gym_make,
    dict(id="FrozenLake-v1", wrap_spaces=True, is_slippery=False),
)
CART_POLE_CONSTR = (gym_make, dict(id="CartPole-v1", wrap_spaces=True))
PENDULUM_CONSTR = (gym_make, dict(id="Pendulum-v1", wrap_spaces=True))
ASTEROIDS_CONSTR = (atari_make, dict(id="ALE/Asteroids-v5", wrap_spaces=True))
CUSTOM_CONSTR = (CustomDummyEnv, {})


ALL_ENVS = [
    FROZEN_LAKE_CONSTR,
    CART_POLE_CONSTR,
    PENDULUM_CONSTR,
    ASTEROIDS_CONSTR,
    CUSTOM_CONSTR,
]


@pytest.mark.parametrize("Env", ALL_ENVS)
def test_env(Env):
    current_env = Env[0](**Env[1])
    if not isinstance(current_env, CustomDummyEnv):
        check_env(current_env)
    check_gym_env(current_env)


A2C_INIT_KWARGS = {"algo_cls": A2C, "policy": "MlpPolicy", "verbose": 1}
DQN_INIT_KWARGS = {"algo_cls": DQN, "policy": "MlpPolicy", "verbose": 1}

AGENTS_WITH_ENV = [
    (A2C_INIT_KWARGS, PENDULUM_CONSTR),
    (DQN_INIT_KWARGS, CART_POLE_CONSTR),
]


@pytest.mark.parametrize("agent_kwargs,env", AGENTS_WITH_ENV)
def test_rlberry_agent(agent_kwargs, env):
    check_rl_agent(
        StableBaselinesAgent,
        env=env,
        init_kwargs=agent_kwargs,
    )
