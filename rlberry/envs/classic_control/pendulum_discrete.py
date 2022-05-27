import gym
import numpy as np
import warnings
from gym import spaces


class PendulumDiscrete:
    def __init__(self):
        self.env = gym.make("Pendulum-v1")
        self.horizon = self.env._max_episode_steps
        self.env._max_episode_steps = np.inf
        self.done_steps = 0
        self.act_mapping = np.asarray([[-2], [0], [2]])
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Discrete(3)

    def reset(self):
        self.done_steps = 0
        return self.env.reset()

    def step(self, act):
        ret = self.env.step(self.act_mapping[act])
        terminal = done = ret[2]
        self.done_steps += 1
        if self.done_steps >= self.horizon:
            done = True
            if terminal:
                warnings.warn(
                    "reached horizon, and terminal is suspiciously True. Check if _max_episode_steps "
                    + "has the expected behavior in your environment"
                )
        return ret[0], ret[1], done, terminal

    def render(self):
        self.env.render()

    def get_nb_act(self):
        return len(self.act_mapping)

    def get_dim_obs(self):
        return self.env.observation_space.shape[0]

    def seed(self, seed):
        self.env.seed(seed)
