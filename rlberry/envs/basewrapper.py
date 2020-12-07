import rlberry.seeding as seeding
import gym
from rlberry.envs.interface import Model


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

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        # If gym environment, reseeding is necessary here for
        # reproducibility.
        if isinstance(env, gym.Env):
            self.reseed()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def spec(self):
        return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

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
        # seed gym.Env that is not a rlberry Model
        if isinstance(self.env, gym.Env) \
                and not isinstance(self.env, Model):
            # get a seed for gym environment
            seed = self.rng.integers(2**16).item()
            self.env.seed(seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        # seed rlberry Model
        else:
            self.env.reseed()
            self.observation_space.rng = self.env.rng
            self.action_space.rng = self.env.rng

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def sample(self, state, action):
        return self.env.sample(state, action)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode=mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

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

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)
