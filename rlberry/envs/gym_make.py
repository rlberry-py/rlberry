import gym
from rlberry.envs.basewrapper import Wrapper


def gym_make(env_name):
    """
    Same as gym.make, but wraps the environment
    to ensure unified seeding with rlberry.
    """
    return Wrapper(gym.make(env_name))
