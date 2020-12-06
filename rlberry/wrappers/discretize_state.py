import numpy as np

import rlberry.spaces as spaces
from rlberry.utils.binsearch import binary_search_nd, unravel_index_uniform_bin
from rlberry.envs import Wrapper


class DiscretizeStateWrapper(Wrapper):
    """
    Discretize an environment with continuous states and discrete actions.
    """

    def __init__(self, _env, n_bins):
        # initialize base class
        super().__init__(_env)

        self.n_bins = n_bins
        # initialize bins
        assert n_bins > 0, "DiscretizeStateWrapper requires n_bins > 0"
        n_states = 1
        tol = 1e-8
        self.dim = len(self.env.observation_space.low)
        n_states = n_bins ** self.dim
        self._bins = []
        self._open_bins = []
        for dd in range(self.dim):
            range_dd = self.env.observation_space.high[dd] \
                - self.env.observation_space.low[dd]
            epsilon = range_dd / n_bins
            bins_dd = []
            for bb in range(n_bins + 1):
                val = self.env.observation_space.low[dd] + epsilon * bb
                bins_dd.append(val)
            self._open_bins.append(tuple(bins_dd[1:]))
            bins_dd[-1] += tol  # "close" the last interval
            self._bins.append(tuple(bins_dd))

            # set observation space
        self.observation_space = spaces.Discrete(n_states)

        # List of discretized states
        self.discretized_states = np.zeros((self.dim, n_states))
        for ii in range(n_states):
            self.discretized_states[:, ii] = \
                self.get_continuous_state(ii, False)

    def reset(self):
        return self.get_discrete_state(self.env.reset())

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = binary_search_nd(next_state, self._bins)
        return next_state, reward, done, info

    def sample(self, discrete_state, action):
        # map disctete state to continuous one
        assert self.observation_space.contains(discrete_state)
        continuous_state = self.get_continuous_state(discrete_state,
                                                     randomize=True)
        # sample in the true environment
        next_state, reward, done, info = \
            self.env.sample(continuous_state, action)
        # discretize next state
        next_state = binary_search_nd(next_state, self._bins)

        return next_state, reward, done, info

    def get_discrete_state(self, continuous_state):
        return binary_search_nd(continuous_state, self._bins)

    def get_continuous_state(self, discrete_state, randomize=False):
        assert discrete_state >= 0 \
            and discrete_state < self.observation_space.n, \
            "invalid discrete_state"
        # get multi-index
        index \
            = unravel_index_uniform_bin(discrete_state, self.dim, self.n_bins)

        # get state
        continuous_state = np.zeros(self.dim)
        for dd in range(self.dim):
            continuous_state[dd] = self._bins[dd][index[dd]]
            if randomize:
                range_dd = self.env.observation_space.high[dd] \
                    - self.env.observation_space.low[dd]
                epsilon = range_dd / self.n_bins
                continuous_state[dd] += epsilon * self.rng.uniform()
        return continuous_state
