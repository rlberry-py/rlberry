import pytest

from rlberry.envs.classic_control import MountainCar, Acrobot
from rlberry.envs.finite import Chain
from rlberry.envs.finite import GridWorld
from rlberry.envs.benchmarks.ball_exploration import PBall2D, SimplePBallND

classes = [
    MountainCar,
    GridWorld,
    Chain,
    PBall2D,
    SimplePBallND,
    Acrobot
]


@pytest.mark.parametrize("ModelClass", classes)
def test_instantiation(ModelClass):
    env = ModelClass()

    if env.is_online():
        for ep in range(2):
            state = env.reset()
            for _ in range(50):
                assert env.observation_space.contains(state)
                action = env.action_space.sample()
                next_s, reward, done, info = env.step(action)
                state = next_s

    if env.is_generative():
        for _ in range(100):
            state = env.observation_space.sample()
            action = env.action_space.sample()
            next_s, reward, done, info = env.sample(state, action)
            assert env.observation_space.contains(next_s)

