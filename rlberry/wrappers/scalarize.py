from rlberry.envs import Wrapper


class ScalarizeEnvWrapper(Wrapper):
    """
    Wrapper for stable_baselines VecEnv, so that they accept non-vectorized actions,
    and return non-vectorized states.
    """

    def __init__(self, env):
        Wrapper.__init__(self, env)

    def reset(self,seed=None,options=None):
        obs,infos = self.env.reset(seed=seed,options=options)
        return obs[0],infos[0]

    def step(self, action):
        observation, reward, done, info = self.env.step(
            [action] * self.env.env.num_envs
        )
        return observation[0], reward[0], done[0], info[0]
