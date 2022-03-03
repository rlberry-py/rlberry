import pytest

import stable_baselines3 as sb3

from rlberry.envs import gym_make
from rlberry.utils.sb_agent import StableBaselinesAgent
from rlberry.utils.check_agent import (
    check_agent_manager,
    check_seeding_agent,
    check_save_load,
)


CONTINUOUS_ACTION_AGENTS = [
    sb3.A2C,
    sb3.DDPG,
    sb3.PPO,
    sb3.SAC,
    sb3.TD3,
]

DISCRETE_ACTION_AGENTS = [
    sb3.A2C,
    sb3.DQN,
    sb3.PPO,
]


def _sb3_check_rl_agent(agent_cls, env, init_kwargs=None):
    """
    Check that an RL agent can be instantiated and fit.

    Note
    ----
    Does not check that the fit is additive, because that doesn't seem to be
    always the case with StableBaselines3.
    """
    check_agent_manager(agent_cls, env, init_kwargs=init_kwargs)
    check_seeding_agent(agent_cls, env, init_kwargs=init_kwargs)
    check_save_load(agent_cls, env, init_kwargs=init_kwargs)


@pytest.mark.parametrize("algo_cls", CONTINUOUS_ACTION_AGENTS)
def test_continuous_action_agent(algo_cls):
    env = gym_make, {"id": "Pendulum-v0"}
    _sb3_check_rl_agent(
        StableBaselinesAgent,
        env=env,
        init_kwargs={"algo_cls": algo_cls, "policy": "MlpPolicy"},
    )


@pytest.mark.parametrize("algo_cls", DISCRETE_ACTION_AGENTS)
def test_discrete_action_agent(algo_cls):
    env = gym_make, {"id": "CartPole-v1"}
    _sb3_check_rl_agent(
        StableBaselinesAgent,
        env=env,
        init_kwargs={"algo_cls": algo_cls, "policy": "MlpPolicy"},
    )
