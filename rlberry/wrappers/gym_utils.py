import gymnasium as gym
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from rlberry.spaces import Discrete
from rlberry.spaces import Box
from rlberry.spaces import Tuple
from rlberry.spaces import MultiDiscrete
from rlberry.spaces import MultiBinary
from rlberry.spaces import Dict

from rlberry.envs import Wrapper


def convert_space_from_gym(gym_space):
    if isinstance(gym_space, gym.spaces.Discrete):
        return Discrete(gym_space.n)
    #
    #
    elif isinstance(gym_space, gym.spaces.Box):
        return Box(gym_space.low, gym_space.high, gym_space.shape, gym_space.dtype)
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




class OldGymCompatibilityWrapper(Wrapper):
    """
    Allow to use old gym env (V0.21) with rlberry (gymnasium).
    (for basic use)
    """

    def __init__(self, env):
        Wrapper.__init__(self, env)

    def reset(self, seed=None, options=None):
        if(seed):
            self.env.reseed(seed)
        observation = self.env.reset()
        return observation,{}

    def step(self, action):
        obs, rewards, terminated, truncated, info = step_api_compatibility(self.env.step(action), output_truncation_bool=True)
        return obs, rewards, terminated, truncated, info


