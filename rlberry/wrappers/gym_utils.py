import rlberry

_GYM_INSTALLED = True
try:
    import gym
except Exception:
    _GYM_INSTALLED = False


def convert_space_from_gym(gym_space):
    global _GYM_INSTALLED
    assert _GYM_INSTALLED, \
        "gym required by convert_space_from_gym() but not installed."

    if isinstance(gym_space, gym.spaces.Discrete):
        n = gym_space.n
        rlberry_space = rlberry.spaces.Discrete(n)
    elif isinstance(gym_space, gym.spaces.Box):
        assert gym_space.high.ndim == 1, \
            "Conversion from gym.spaces.Box requires high and low to be 1d."
        high = gym_space.high
        low = gym_space.low
        rlberry_space = rlberry.spaces.Box(low, high)

    return rlberry_space
