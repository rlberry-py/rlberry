import rlberry.seeding as seeding
import logging

from rlberry.envs.interface import Model
from rlberry.wrappers.gym_utils import convert_space_from_gym


_GYM_INSTALLED = True
try:
    import gym
except Exception:
    _GYM_INSTALLED = False


class Wrapper(Model):
    """
    Wraps a given environment, similar to OpenAI gym's wrapper [1].
    Can also be used to wrap gym environments.

    Note:
        The input environment is not copied (Wrapper.env points
        to the input env).

    See also:
    https://stackoverflow.com/questions/1443129/completely-wrap-an-object-in-python

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
                'GymWrapper: Rendering gym.Env does not \
follow the same protocol as rlberry.')
            # Convert spaces
            self.observation_space = convert_space_from_gym(
                gym_env.observation_space)
            self.action_space = convert_space_from_gym(gym_env.action_space)
            # Reward range
            self.reward_range = gym_env.reward_range

        # If rlberry environment
        else:
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self.reward_range = self.env.reward_range

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, attr):
        """
        The first condition is to avoid infinite recursion when deep copying.
        See https://stackoverflow.com/a/47300262
        """
        if attr[:2] == '__':
            raise AttributeError(attr)
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.env, attr)

    def reseed(self):
        self.rng = seeding.get_rng()
        if _GYM_INSTALLED and isinstance(self.env, gym.Env):
            # get a seed for gym environment
            self.env.seed(self.rng.integers(2**16).item())
        else:
            self.env.reseed()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def sample(self, state, action):
        return self.env.sample(state, action)

    def is_online(self):
        try:
            self.env.reset()
            self.env.step(self.env.action_space.sample())
            return True
        except Exception:
            return False

    def is_generative(self):
        try:
            self.env.sample(self.env.observation_space.sample(),
                            self.env.action_space.sample())
            return True
        except Exception:
            return False

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)
