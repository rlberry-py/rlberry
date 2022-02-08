from gym.spaces import Box, Discrete
import rlberry.agents.torch.td3.td3_utils as utils
from rlberry.envs import Wrapper
import numpy as np


class NormalizedContinuousEnvWrapper(Wrapper):
    """
    Wraps the observation and action spaces of environments
    so that they can be used in TD3.
    """

    def __init__(self, env):
        Wrapper.__init__(self, env)
        obs_space = self.env.observation_space
        action_space = self.env.action_space
        assert isinstance(obs_space, Box) or isinstance(obs_space, Discrete)
        assert isinstance(action_space, Box) or isinstance(action_space, Discrete)

        self.make_one_hot_obs = isinstance(obs_space, Discrete)
        if isinstance(action_space, Discrete):
            self.action_is_prob = True
            self.rescale_actions = False
        else:
            assert action_space.is_bounded()
            self.action_is_prob = False
            self.rescale_actions = True

        # Set new spaces
        if self.make_one_hot_obs:
            self.observation_space = Box(
                low=0.0, high=1.0, shape=(obs_space.n,), dtype=np.uint32
            )
        else:
            self.observation_space = obs_space

        if self.action_is_prob:
            self.action_space = Box(
                low=0.0, high=1.0, shape=(action_space.n,), dtype=np.float32
            )
        else:
            self.action_space = Box(
                low=-1.0, high=1.0, shape=action_space.shape, dtype=np.float32
            )

    def process_obs(self, obs):
        if self.make_one_hot_obs:
            one_hot_obs = np.zeros(self.env.observation_space.n, dtype=np.uint32)
            one_hot_obs[obs] = 1.0
            return one_hot_obs
        else:
            return np.array(obs, dtype=np.float32)

    def process_action(self, action):
        if self.action_is_prob:
            # normalize action, if not normalized
            sum_action = np.sum(action)
            if sum_action > 0.0 and sum_action != 1.0:
                action = action / sum_action
            elif sum_action == 0:
                action = np.ones_like(action) / len(action)
            return self.rng.choice(self.env.action_space.n, p=action)
        else:
            # action is in [-1, 1], map to [low, high]
            return utils.unscale_action(self.env.action_space, action)

    def reset(self):
        obs = self.env.reset()
        return self.process_obs(obs)

    def step(self, action):
        action = self.process_action(action)
        observation, reward, done, info = self.env.step(action)
        observation = self.process_obs(observation)
        return observation, reward, done, info


if __name__ == "__main__":
    import gym
    from rlberry.envs.finite import GridWorld

    for env in [gym.make("CartPole-v0"), gym.make("Pendulum-v1"), GridWorld()]:
        wrapped_env = NormalizedContinuousEnvWrapper(env)

        wrapped_env.reset()
        for _ in range(10):
            action = wrapped_env.action_space.sample()
            print(wrapped_env.step(action))
        print("")
        print(wrapped_env.observation_space.sample())
        print("---")
