import pytest 
from rlberry.envs.interface import Model
from rlberry.wrappers import Wrapper
from rlberry.envs import GridWorld, MountainCar
from rlberry.spaces import Discrete, Box

_GYM_INSTALLED = True
try:
    import gym
except:
    _GYM_INSTALLED = False


def test_wrapper():
    global _GYM_INSTALLED 

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
    if _GYM_INSTALLED:
        gym_env = gym.make('Acrobot-v1')
        wrapped = Wrapper(gym_env)
        assert isinstance(wrapped, Model)
        assert wrapped.is_online()
        assert not wrapped.is_generative()
        
        assert isinstance(wrapped.observation_space, Box)
        assert isinstance(wrapped.action_space, Discrete)
        
        # calling some gym functions
        wrapped.close()
        wrapped.seed()


    
