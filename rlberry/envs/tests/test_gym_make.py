from rlberry.envs.gym_make import atari_make
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, MaxAndSkipEnv


def test_atari_make():
    wrappers_dict = dict(terminal_on_life_loss=True, frame_skip=8)
    env = atari_make(
        "ALE/Freeway-v5", render_mode="rgb_array", atari_wrappers_dict=wrappers_dict
    )
    assert "EpisodicLifeEnv" in str(env.unwrapped.envs[0])
    assert "MaxAndSkipEnv" in str(env.unwrapped.envs[0])
    assert env.render_mode == "rgb_array"

    wrappers_dict2 = dict(terminal_on_life_loss=False, frame_skip=0)
    env2 = atari_make(
        "ALE/Breakout-v5", render_mode="human", atari_wrappers_dict=wrappers_dict2
    )
    assert "EpisodicLifeEnv" not in str(env2.unwrapped.envs[0])
    assert "MaxAndSkipEnv" not in str(env2.unwrapped.envs[0])
    assert "MaxAndSkipEnv" not in str(env2.unwrapped.envs[0])
    assert env2.render_mode == "human"
