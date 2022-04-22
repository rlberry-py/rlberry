import tempfile

from stable_baselines3 import A2C

from rlberry.envs import gym_make
from rlberry.agents import StableBaselinesAgent
from rlberry.utils.check_agent import check_rl_agent


def test_sb3_agent():
    # Test only one algorithm per action space type
    check_rl_agent(
        StableBaselinesAgent,
        env=(gym_make, {"id": "Pendulum-v1"}),
        init_kwargs={"algo_cls": A2C, "policy": "MlpPolicy", "verbose": 1},
    )
    check_rl_agent(
        StableBaselinesAgent,
        env=(gym_make, {"id": "CartPole-v1"}),
        init_kwargs={"algo_cls": A2C, "policy": "MlpPolicy", "verbose": 1},
    )


def test_sb3_tensorboard_log():
    # Test tensorboard support
    with tempfile.TemporaryDirectory() as tmpdir:
        check_rl_agent(
            StableBaselinesAgent,
            env=(gym_make, {"id": "Pendulum-v1"}),
            init_kwargs={
                "algo_cls": A2C,
                "policy": "MlpPolicy",
                "verbose": 1,
                "tensorboard_log": tmpdir,
            },
        )
