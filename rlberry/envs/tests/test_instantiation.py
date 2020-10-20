import pytest 

from rlberry.envs.interface import ForwardModel, GenerativeModel

from rlberry.envs.classic_control import MountainCar 

from rlberry.envs.finite import GridWorld
from rlberry.envs.finite import Chain

from rlberry.envs.toy_exploration import PBall2D
from rlberry.envs.toy_exploration import SimplePBallND


classes = [
            MountainCar,
            GridWorld,
            Chain,
            PBall2D,
            SimplePBallND
          ]

@pytest.mark.parametrize("ModelClass", classes)
def test_instantiation(ModelClass):
    env = ModelClass()

    if isinstance(env, ForwardModel):
        for ep in range(2):
            state = env.reset()
            for ii in range(50):
                assert env.observation_space.contains(state)
                action = env.action_space.sample()
                next_s, reward, done, info = env.step(action)
                state = next_s 

    if isinstance(env, GenerativeModel):
        for ii in range(100):
            state  = env.observation_space.sample()
            action = env.action_space.sample()
            next_s, reward, done, info = env.sample(state, action)
            assert env.observation_space.contains(next_s)
