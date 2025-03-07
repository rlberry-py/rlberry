import gymnasium as gym
from rlberry.wrappers.utils import get_base_env


def test_get_base_env():
    """Test that the utils function 'get_base_env' return the wrapped env without the wrappers"""

    from rlberry.envs.basewrapper import Wrapper
    from stable_baselines3.common.monitor import Monitor

    from stable_baselines3.common.atari_wrappers import (  # isort:skip
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
        NoopResetEnv,
        StickyActionEnv,
    )

    env = gym.make("ALE/Breakout-v5")
    original_env = env

    # add wrappers
    env = Wrapper(env)
    env = Monitor(env)
    env = StickyActionEnv(env, 0.2)
    env = NoopResetEnv(env, noop_max=2)
    env = MaxAndSkipEnv(env, skip=4)
    env = FireResetEnv(env)
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 8)
    assert original_env != env

    # use the tool
    unwrapped_env = get_base_env(env)

    # test the result
    assert unwrapped_env != env
    assert isinstance(unwrapped_env, gym.Env)
    assert unwrapped_env == get_base_env(original_env)
