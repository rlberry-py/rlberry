# import gym
# from rlberry.envs.basewrapper import Wrapper
from rlberry.envs.classic_control.pendulum_discrete import PendulumDiscrete


def make_pendulum(id, wrap_spaces=False, **kwargs):
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
    """
    if "module_import" in kwargs:
        __import__(kwargs.pop("module_import"))
    env = PendulumDiscrete()
    try:
        env.configure(kwargs)
    except AttributeError:
        pass
    return env
