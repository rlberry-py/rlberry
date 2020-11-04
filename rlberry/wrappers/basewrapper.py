from copy import deepcopy
from rlberry.envs.interface import Model, GenerativeModel, OnlineModel, SimulationModel
from rlberry.wrappers.gym_utils import convert_space_from_gym

import rlberry.seeding as seeding

import logging

_GYM_INSTALLED = True
try:
    import gym
except:
    _GYM_INSTALLED = False


class Wrapper(Model):
    """
    Wraps a given environment, similar to OpenAI gym's wrapper [1].
    Can also be used to wrap gym environments.

    The type of the wrapper is defined in runtime, according to the type of the 
    wrapped environment (gym.Env, OnlineModel, GenerativeModel or SimulationModel)

    Note: 
        The input environment is not copied (Wrapper.env points to the input env).
        
    See also: https://stackoverflow.com/questions/1443129/completely-wrap-an-object-in-python

    [1] https://github.com/openai/gym/blob/master/gym/core.py
    """
    def __init__(self, env):
        # Init base class
        Model.__init__(self)

        # Save reference to env
        self.env = env

        # Check if gym environment
        if _GYM_INSTALLED and isinstance(env, gym.Env):
            gym_env = env 
            # Warnings
            logging.warning(
                'GymWrapper: Seeding of gym.Env does not follow the same protocol as rlberry. Make sure to properly seed each instance before using the wrapped environment.')
            logging.warning(
                'GymWrapper: Rendering gym.Env does not follow the same protocol as rlberry.')
            # Convert spaces
            self.observation_space = convert_space_from_gym(
                gym_env.observation_space)
            self.action_space = convert_space_from_gym(gym_env.action_space)
            # Reward range
            self.reward_range = gym_env.reward_range

            # Set class
            self.__class__ = type(self.__class__.__name__,
                              (self.__class__, OnlineModel), # gym environment is a rlberry OnlineModel
                              self.__dict__)

        # If rlberry environment
        else:
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self.reward_range = self.env.reward_range 
            # Set class
            wrapper_class = Model 
            if isinstance(env, SimulationModel):
                wrapper_class = SimulationModel 
            elif isinstance(env, OnlineModel):
                wrapper_class = OnlineModel 
            elif isinstance(env, GenerativeModel):
                wrapper_class = GenerativeModel 
            self.__class__ = type(self.__class__.__name__,
                            (self.__class__, wrapper_class),
                            self.__dict__) 

    @property
    def unwrapped(self):
        return self.env

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.env, attr)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def sample(self, state, action):
        return self.env.sample(state, action)

