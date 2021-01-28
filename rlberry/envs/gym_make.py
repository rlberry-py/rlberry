from typing import Optional

import gym
from rlberry.envs.basewrapper import Wrapper


def gym_make(env_name, **kwargs):
    """
    Same as gym.make, but wraps the environment
    to ensure unified seeding with rlberry.
    """
    if "module_import" in kwargs:
        __import__(kwargs.pop("module_import"))
    env = gym.make(env_name)
    try:
        env.configure(kwargs)
    except AttributeError:
        pass
    return Wrapper(env)


def atari_make(env_name, scalarize=True, **kwargs):
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack
    env = make_atari_env(env_id=env_name, **kwargs)
    env = VecFrameStack(env, n_stack=4)
    if scalarize:
        from rlberry.wrappers.scalarize import ScalarizeEnvWrapper
        env = ScalarizeEnvWrapper(env)
    return env
