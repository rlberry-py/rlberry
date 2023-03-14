import gymnasium as gym
# from gymnasium.wrappers import StepAPICompatibility

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

    # if old_gym:
    #     env = gym.make(id, **kwargs)
    #     env = StepAPICompatibility(env,output_truncation_bool=False)
    # else:
    env = gym.make(id, **kwargs)

    return Wrapper(env, wrap_spaces=wrap_spaces)


def atari_make(id, scalarize=None, **kwargs):
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack

    # #uncomment when rlberry will manage vectorized env
    # if scalarize is None:
    #     if "n_envs" in kwargs.keys() and int(kwargs["n_envs"])>1:
    #         scalarize = False
    #     else:
    #         scalarize = True

    scalarize = True

    if "atari_wrappers_dict" in kwargs.keys():
        atari_wrappers_dict = kwargs["atari_wrappers_dict"]
        kwargs.pop("atari_wrappers_dict", None)
    else:
        atari_wrappers_dict = dict(
            terminal_on_life_loss=False
        )  # hack, some errors with the "terminal_on_life_loss" wrapper : The 'false reset' can lead to make a step on a 'done' environment, then a crash.

    env = make_atari_env(env_id=id, wrapper_kwargs=atari_wrappers_dict, **kwargs)

    env = VecFrameStack(env, n_stack=4)
    env = SB3_AtariImageToPyTorch(env)
    if scalarize:
        from rlberry.wrappers.scalarize import ScalarizeEnvWrapper

        env = ScalarizeEnvWrapper(env)

    return env


class SB3_AtariImageToPyTorch(Wrapper):
    """
    transform the observations shape.
    from: n_env, height, width, chan
    to: n_env, chan, width, height

    WARNING : Check the Reset and Step format :
    https://github.com/DLR-RM/stable-baselines3/pull/1327/files#diff-a0b0c17357564df74e097f3094a5478e9b28b2af9dfdab2a91e60b6dbe174092

    """


    def __init__(self, env):
        super(SB3_AtariImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )

    def observation(self, observation):
        return np.transpose(observation, (0, 3, 2, 1))  # transform

    def reset(self):
        obs = self.env.reset()
        infos = self.env.reset_infos
        return self.observation(obs), infos

    def step(self, actions):
        next_observations, rewards, done, infos = self.env.step(actions)
        # return self.observation(next_observations), rewards, done, [d["TimeLimit.truncated"] for d in infos], infos
        return self.observation(next_observations), rewards, done, [None]*self.env.num_envs, infos
