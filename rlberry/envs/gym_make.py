import gym

from rlberry.envs.basewrapper import Wrapper
import numpy as np


def gym_make(id, wrap_spaces=False, **kwargs):
    """
    Same as gym.make, but wraps the environment
    to ensure unified seeding with rlberry.

    Parameters
    ----------
    id : str
        Environment id.
    wrap_spaces : bool, default = False
        If true, also wraps observation_space and action_space using classes in rlberry.spaces,
        that define a reseed() method.
    **kwargs
        Optional arguments to configure the environment.

    Examples
    --------
    >>> from rlberry.envs import gym_make
    >>> env_ctor = gym_make
    >>> env_kwargs = {"id": "CartPole-v0"}
    >>> env = env_ctor(**env_kwargs)
    """
    if "module_import" in kwargs:
        __import__(kwargs.pop("module_import"))

    env = gym.make(id, **kwargs)
    return Wrapper(env, wrap_spaces=wrap_spaces)


def atari_make(id, scalarize=False, **kwargs):
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack

    # #uncomment when rlberry will manage vectorized env
    # if scalarize is None:
    #     if "n_envs" in kwargs.keys() and int(kwargs["n_envs"])>1:
    #         scalarize = False
    #     else:
    #         scalarize = True

    scalarize = True    #TODO : to remove with th PR :[WIP] Atari part2   (https://github.com/rlberry-py/rlberry/pull/285)

    if "atari_wrappers_dict" in kwargs.keys():
        atari_wrappers_dict = kwargs.pop("atari_wrappers_dict")
    else:
        atari_wrappers_dict = dict(
            terminal_on_life_loss=False
        )  # hack, some errors with the "terminal_on_life_loss" wrapper : The 'false reset' can lead to make a step on a 'done' environment, then a crash.

    env = make_atari_env(env_id=id, wrapper_kwargs=atari_wrappers_dict, **kwargs)

    env = VecFrameStack(env, n_stack=4)
    env = AtariImageToPyTorch(env)
    if scalarize:
        from rlberry.wrappers.scalarize import ScalarizeEnvWrapper

        env = ScalarizeEnvWrapper(env)
    return env


class AtariImageToPyTorch(Wrapper):
    """
    #transform the observation shape.
    from: n_env, height, width, chan
    to: n_env, chan, width, height
    """

    def __init__(self, env):
        super(AtariImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )

    def observation(self, observation):
        return np.transpose(observation, (0, 3, 2, 1))  # transform

    def reset(self):
        return self.observation(self.env.reset())

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return self.observation(next_obs), reward, done, info
