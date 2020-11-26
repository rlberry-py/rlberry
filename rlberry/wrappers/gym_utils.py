import gym
from rlberry.spaces import Discrete
from rlberry.spaces import Box
from rlberry.spaces import Tuple
from rlberry.spaces import MultiDiscrete
from rlberry.spaces import MultiBinary
from rlberry.spaces import Dict


def convert_space_from_gym(gym_space):
    if isinstance(gym_space, gym.spaces.Discrete):
        return Discrete(gym_space.n)
    #
    #
    elif isinstance(gym_space, gym.spaces.Box):
        return Box(gym_space.low,
                   gym_space.high,
                   gym_space.shape,
                   gym_space.dtype)
    #
    #
    elif isinstance(gym_space, gym.spaces.Tuple):
        spaces = []
        for sp in gym_space.spaces:
            spaces.append(convert_space_from_gym(sp))
        return Tuple(spaces)
    #
    #
    elif isinstance(gym_space, gym.spaces.MultiDiscrete):
        return MultiDiscrete(gym_space.nvec)
    #
    #
    elif isinstance(gym_space, gym.spaces.MultiBinary):
        return MultiBinary(gym_space.n)
    #
    #
    elif isinstance(gym_space, gym.spaces.Dict):
        spaces = {}
        for key in gym_space.spaces:
            spaces[key] = convert_space_from_gym(gym_space[key])
        return Dict(spaces)
    else:
        raise ValueError("Unknown space class: {}".format(type(gym_space)))
