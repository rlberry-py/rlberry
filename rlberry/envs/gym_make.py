import gym

_GYMNASIUM_INSTALLED = True
try:
    import gymnasium
except Exception:
    _GYMNASIUM_INSTALLED = False


from rlberry.envs.basewrapper import Wrapper


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
    env = gym.make(id)
    try:
        env.configure(kwargs)
    except AttributeError:
        pass
    return Wrapper(env, wrap_spaces=wrap_spaces)


def gymnasium_make(id, wrap_spaces=False, **kwargs):
    """
    Same as gymnasium.make, but wraps the environment
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
    assert _GYMNASIUM_INSTALLED, "module not found: gymnasium"
    if "module_import" in kwargs:
        __import__(kwargs.pop("module_import"))
    env = gymnasium.make(id)
    try:
        env.configure(kwargs)
    except AttributeError:
        pass
    return Wrapper(env, wrap_spaces=wrap_spaces)


def atari_make(id, scalarize=True, **kwargs):
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack

    env = make_atari_env(env_id=id, **kwargs)
    env = VecFrameStack(env, n_stack=4)
    if scalarize:
        from rlberry.wrappers.scalarize import ScalarizeEnvWrapper

        env = ScalarizeEnvWrapper(env)
    return env
