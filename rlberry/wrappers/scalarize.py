from rlberry.envs import Wrapper


class ScalarizeEnvWrapper(Wrapper):
    """
    Wrapper for stable_baselines VecEnv, so that they accept non-vectorized actions,
    and return non-vectorized states.
    """

    def __init__(self, env):
        Wrapper.__init__(self, env)

    def reset(self):
        obs = self.env.reset()
        return obs[0]

    def step(self, action):
        observation, reward, done, info = self.env.step([action])
        return observation[0], reward[0], done[0], info[0]
