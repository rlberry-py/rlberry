import gymnasium as gym
from rlberry.seeding import Seeder, safe_reseed
import numpy as np
from rlberry.envs.interface import Model
from rlberry.rendering import RenderInterface
from rlberry.spaces.from_gym import convert_space_from_gym
from rlberry.rendering.utils import video_write, gif_write


class Wrapper(Model, RenderInterface):
    """
    Wraps a given environment, similar to OpenAI gym's wrapper [1,2] (now updated to gymnasium).
    Can also be used to wrap gym environments.

    Note:
        The input environment is not copied (Wrapper.env points
        to the input env).

    Parameters
    ----------
    env: gymnasium.Env
        Environment to be wrapped.
    wrap_spaces: bool, default = False
        If True, gymnasium.spaces are converted to rlberry.spaces, which defined a reseed() method.

    Attributes
    ----------
    env : gymnasium.Env
        The wrapped environment
    metadata : dict
        InitiallThe meatadata of the wrapped environment
    render_mode : str
        The render_mode of the wrapped environment


    See also:
    https://stackoverflow.com/questions/1443129/completely-wrap-an-object-in-python

    [1] https://github.com/openai/gym/blob/master/gym/core.py
    [2] https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py
    """

    def __init__(self, env, wrap_spaces=False):
        # Init base class
        Model.__init__(self)

        # Save reference to env
        self.env = env
        self.metadata = self.env.metadata
        self.render_mode = self.env.render_mode
        self.frames = []

        if wrap_spaces:
            self.observation_space = convert_space_from_gym(self.env.observation_space)
            self.action_space = convert_space_from_gym(self.env.action_space)
        else:
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space

        try:
            self.reward_range = self.env.reward_range
        except AttributeError:
            self.reward_range = (-np.inf, np.inf)

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
        if attr[:2] == "__":
            raise AttributeError(attr)
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.env, attr)

    def reseed(self, seed_seq=None):
        # self.seeder
        if seed_seq is None:
            self.seeder = self.seeder.spawn()
        else:
            self.seeder = Seeder(seed_seq)

        # get a seed for gym environment; spaces are reseeded below.
        if isinstance(self.env, Model):
            # seed rlberry Model
            self.env.reseed(self.seeder)
        elif isinstance(self.env, gym.Env):
            # seed gym.Env that is not a rlberry Model
            seed_val = self.seeder.rng.integers(2**32).item()
            self.env.reset(seed=seed_val)
        else:
            # other
            safe_reseed(self.env, self.seeder, reseed_spaces=False)

        safe_reseed(self.observation_space, self.seeder)
        safe_reseed(self.action_space, self.seeder)

    def reset(self, seed=None, options=None):
        if self.env.render_mode == "human":
            self.render()
        self.frames = []
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if self.render_mode == "human":
            self.render()
        elif self.render_mode == "rgb_array":
            self.frames.append(self.render())
        return self.env.step(action)

    def sample(self, state, action):
        return self.env.sample(state, action)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        # return self.env.seed(seed)
        return self.env.reset(seed=seed)

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
            self.env.sample(
                self.env.observation_space.sample(), self.env.action_space.sample()
            )
            return True
        except Exception:
            return False

    def get_video(self, **kwargs):
        return self.frames

    def save_video(self, filename, framerate=25, **kwargs):
        video_data = self.get_video(**kwargs)
        video_write(filename, video_data, framerate=framerate)

    def save_gif(self, filename, **kwargs):
        video_data = self.get_video(**kwargs)
        gif_write(filename, video_data)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<{}{}>".format(type(self).__name__, self.env)
