import numpy as np
from rlberry.wrappers import Wrapper


class AutoResetWrapper(Wrapper):
    """
    Auto reset the environment after "horizon" steps have passed.
    """
    def __init__(self, env, horizon):
        """
        Parameters
        ----------
        horizon: int
        """
        Wrapper.__init__(self, env)
        self.horizon = horizon
        assert self.horizon >= 1
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.current_step += 1
        if self.current_step == self.horizon:  # At H, always return to the initial state.
            self.current_step = 0
            observation = self.env.reset()
        return observation, reward, done, info
