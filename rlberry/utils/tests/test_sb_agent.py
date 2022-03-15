import tempfile

from stable_baselines3 import A2C

from rlberry.envs import gym_make
from rlberry.utils.sb_agent import StableBaselinesAgent
from rlberry.utils.check_agent import (
    check_agent_manager,
    check_seeding_agent,
    check_save_load,
)


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


def test_sb3_agent():
    # Test only one algorithm per action space type
    _sb3_check_rl_agent(
        StableBaselinesAgent,
        env=(gym_make, {"id": "Pendulum-v0"}),
        init_kwargs={"algo_cls": A2C, "policy": "MlpPolicy", "verbose": 1},
    )
    _sb3_check_rl_agent(
        StableBaselinesAgent,
        env=(gym_make, {"id": "CartPole-v1"}),
        init_kwargs={"algo_cls": A2C, "policy": "MlpPolicy", "verbose": 1},
    )


def test_sb3_tensorboard_log():
    # Test tensorboard support
    with tempfile.TemporaryDirectory() as tmpdir:
        _sb3_check_rl_agent(
            StableBaselinesAgent,
            env=(gym_make, {"id": "Pendulum-v0"}),
            init_kwargs={
                "algo_cls": A2C,
                "policy": "MlpPolicy",
                "verbose": 1,
                "tensorboard_log": tmpdir,
            },
        )
