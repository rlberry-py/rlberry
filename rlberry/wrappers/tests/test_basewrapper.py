import gymnasium as gym

from rlberry.envs import GridWorld, Wrapper
from rlberry.envs.interface import Model


def test_wrapper():
    env = GridWorld()
    wrapped = Wrapper(env)
    assert isinstance(wrapped, Model)
    assert wrapped.is_online()
    assert wrapped.is_generative()

    # calling some functions
    wrapped.reset()
    wrapped.step(wrapped.action_space.sample())
    wrapped.sample(wrapped.observation_space.sample(), wrapped.action_space.sample())


def test_gym_wrapper():
    gym_env = gym.make("Acrobot-v1")
    wrapped = Wrapper(gym_env)
    assert isinstance(wrapped, Model)
    assert wrapped.is_online()
    assert not wrapped.is_generative()

    wrapped.reseed()

    # calling some gym functions
    wrapped.close()
    wrapped.seed()
