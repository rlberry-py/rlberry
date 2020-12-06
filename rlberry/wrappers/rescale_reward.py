import numpy as np
from rlberry.envs import Wrapper


class RescaleRewardWrapper(Wrapper):
    """
    Rescale the reward function to a bounded range.

    Parameters
    ----------
    reward_range: tuple (double, double)
        tuple with the desired reward range, which needs to be bounded.
    """

    def __init__(self, env, reward_range):
        Wrapper.__init__(self, env)
        self.reward_range = reward_range
        assert reward_range[0] < reward_range[1]
        assert reward_range[0] > -np.inf and reward_range[1] < np.inf

    def _linear_rescaling(self, x, x0, x1, u0, u1):
        """
        For x a value in [x0, x1], maps x linearly to the interval [u0, u1].
        """
        a = (u1 - u0) / (x1 - x0)
        b = (x1 * u0 - x0 * u1) / (x1 - x0)
        return a * x + b

    def _rescale(self, reward):
        x0, x1 = self.env.reward_range
        u0, u1 = self.reward_range
        # bounded reward
        if x0 > -np.inf and x1 < np.inf:
            return self._linear_rescaling(reward, x0, x1, u0, u1)
        # unbounded
        elif x0 > -np.inf and x1 == np.inf:
            x = reward - x0  # [0, infty]
            x = 2.0 / (1.0 + np.exp(-x)) - 1.0  # [0, 1]
            return self._linear_rescaling(x, 0.0, 1.0, u0, u1)
            # unbouded below
        elif x0 == -np.inf and x1 < np.inf:
            x = reward - x1  # [-infty, 0]
            x = 2.0 / (1.0 + np.exp(-x))  # [0, 1]
            return self._linear_rescaling(x, 0.0, 1.0, u0, u1)
            # unbounded
        else:
            x = 1.0 / (1.0 + np.exp(-reward))  # [0, 1]
            return self._linear_rescaling(x, 0.0, 1.0, u0, u1)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        rescaled_reward = self._rescale(reward)
        return observation, rescaled_reward, done, info

    def sample(self, state, action):
        observation, reward, done, info = self.env.sample(state, action)
        rescaled_reward = self._rescale(reward)
        return observation, rescaled_reward, done, info
