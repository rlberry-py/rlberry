from rlberry.spaces import Box, Discrete
from rlberry.envs import Wrapper
import numpy as np


class DiscreteToOneHotWrapper(Wrapper):
    """Converts observation spaces from Discrete to Box via one-hot encoding."""

    def __init__(self, env):
        Wrapper.__init__(self, env, wrap_spaces=True)
        obs_space = self.env.observation_space
        assert isinstance(obs_space, Discrete)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(obs_space.n,), dtype=np.uint32
        )

    def process_obs(self, obs):
        one_hot_obs = np.zeros(self.env.observation_space.n, dtype=np.uint32)
        one_hot_obs[obs] = 1.0
        return one_hot_obs

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.process_obs(obs), info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.process_obs(observation)
        return observation, reward, done, info
