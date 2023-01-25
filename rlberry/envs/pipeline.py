from rlberry.envs import Wrapper


class PipelineEnv(Wrapper):
    """
    Environment defined as a pipeline of wrappers and an environment to wrap.

    Parameters
    ----------
    env_ctor: environment class

    env_kwargs: dictionary
        kwargs fed to the environment

    wrappers: list of tuple (wrapper, wrapper_kwargs)
        list of tuple (wrapper, wrapper_kwargs) to be applied to the environment.
        The list [wrapper1, wrapper2] will be applied in the order wrapper1(wrapper2(env))

    Examples
    --------
    >>> from rlberry.envs import PipelineEnv
    >>> from rlberry.envs import gym_make
    >>> from rlberry.wrappers import RescaleRewardWrapper
    >>>
    >>> env_ctor, env_kwargs = PipelineEnv, {
    >>>     "env_ctor": gym_make,
    >>>     "env_kwargs": {"id": "Acrobot-v1"},
    >>>     "wrappers": [(RescaleRewardWrapper, {"reward_range": (0, 1)})],
    >>> }
    >>> eval_env = (gym_make, {"id":"Acrobot-v1"}) # unscaled env for evaluation

    """

    def __init__(self, env_ctor, env_kwargs, wrappers):
        env = env_ctor(**env_kwargs)
        for wrapper in wrappers[::-1]:
            env = wrapper[0](env, **wrapper[1])
        env.reset()
        Wrapper.__init__(self, env)
