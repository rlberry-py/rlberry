import gym
from rlberry.envs.basewrapper import Wrapper


def gym_make(env_name, **kwargs):
    """
    Same as gym.make, but wraps the environment
    to ensure unified seeding with rlberry.
    """
    env = gym.make(env_name)
    try:
        env.configure(kwargs)
        env.reset()
    except AttributeError:
        pass
    return Wrapper(env)


def atari_make(env_name, **kwargs):
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack
    env = make_atari_env(env_id=env_name, **kwargs)
    env = VecFrameStack(env, n_stack=4)
    return env


