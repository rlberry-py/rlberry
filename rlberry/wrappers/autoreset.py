from rlberry.envs import Wrapper


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

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        # At H, always return to the initial state.
        # Also, set done to True.
        if self.current_step == self.horizon:
            self.current_step = 0
            observation, info = self.env.reset()
            terminated = True
            truncated = False
        return observation, reward, terminated, truncated, info
