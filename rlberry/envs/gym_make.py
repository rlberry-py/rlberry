import gymnasium as gym

from rlberry.envs.basewrapper import Wrapper
import numpy as np
from numpy import ndarray


# VERSION_ORIGINE = True
VERSION_ORIGINE = False


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


def atari_make(id, seed=None, **kwargs):
    """
    Adaptator to manage Atari Env

    Parameters
    ----------
    id : str
        Environment id.
    **kwargs
        Optional arguments to configure the environment.
        (render_mode, n_frame_stack, and other arguments for StableBaselines's make_atari_env : https://stable-baselines3.readthedocs.io/en/master/common/env_util.html#stable_baselines3.common.env_util.make_atari_env )
    Returns
    -------
    Atari env with wrapper to be used as Gymnasium env.
    Examples
    --------
    >>> from rlberry.envs.gym_make import atari_make
    >>> env_ctor = atari_make
    >>> env_kwargs = {"id": "ALE/Freeway-v5", "atari_wrappers_dict":dict(terminal_on_life_loss=False),"n_frame_stack":5}}
    >>> env = env_ctor(**env_kwargs)
    """

    from stable_baselines3.common.atari_wrappers import (  # isort:skip
        ClipRewardEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
        NoopResetEnv,
        StickyActionEnv,
    )

    from stable_baselines3.common.monitor import Monitor

    # Default values for Atari_SB3_wrappers
    noop_max = 30
    frame_skip = 4
    screen_size = 84
    terminal_on_life_loss = False  # different from SB3 : some errors with the "terminal_on_life_loss" wrapper : The 'false reset' can lead to make a step on a 'done' environment, then a crash.
    clip_reward = True
    action_repeat_probability = 0.0

    if "atari_SB3_wrappers_dict" in kwargs.keys():
        atari_wrappers_dict = kwargs.pop("atari_SB3_wrappers_dict")
        if "noop_max" in atari_wrappers_dict.keys():
            noop_max = atari_wrappers_dict["noop_max"]
        if "frame_skip" in atari_wrappers_dict.keys():
            frame_skip = atari_wrappers_dict["frame_skip"]
        if "screen_size" in atari_wrappers_dict.keys():
            screen_size = atari_wrappers_dict["screen_size"]
        if "terminal_on_life_loss" in atari_wrappers_dict.keys():
            terminal_on_life_loss = atari_wrappers_dict["terminal_on_life_loss"]
        if "clip_reward" in atari_wrappers_dict.keys():
            clip_reward = atari_wrappers_dict["clip_reward"]
        if "action_repeat_probability" in atari_wrappers_dict.keys():
            action_repeat_probability = atari_wrappers_dict["action_repeat_probability"]

    render_mode = None
    if "render_mode" in kwargs.keys():
        render_mode = kwargs["render_mode"]
        kwargs.pop("render_mode", None)

    if "n_frame_stack" in kwargs.keys():
        n_frame_stack = kwargs.pop("n_frame_stack")
    else:
        n_frame_stack = 4

    env = gym.make(id)
    env = Wrapper(env)
    env = Monitor(env)

    if action_repeat_probability > 0.0:
        env = StickyActionEnv(env, action_repeat_probability)
    if noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    if frame_skip > 1:
        env = MaxAndSkipEnv(env, skip=frame_skip)
    if terminal_on_life_loss:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if clip_reward:
        env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (screen_size, screen_size))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, n_frame_stack)
    if seed:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    env = CompatibleWrapper(env)  # Wrapper to make it compatible with rlberry

    env.render_mode = render_mode
    return env


class CompatibleWrapper(Wrapper):
    def __init__(self, env):
        super(CompatibleWrapper, self).__init__(env)
        self.render_mode = None

    def step(self, action):
        if type(action) is ndarray and action.size == 1:
            action = action[0]

        next_observations, rewards, terminated, truncated, infos = self.env.step(action)
        return (
            np.array(next_observations),
            rewards,
            terminated,
            truncated,
            infos,
        )

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)

        return np.array(obs), infos
