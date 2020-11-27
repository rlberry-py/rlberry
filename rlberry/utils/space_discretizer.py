import numpy as np
from gym.spaces import Box, Discrete
from rlberry.utils.binsearch import binary_search_nd
from rlberry.utils.binsearch import unravel_index_uniform_bin


class Discretizer:
    def __init__(self, space, n_bins):
        assert isinstance(space, Box), \
            "Discretization is only implemented for Box spaces."
        assert space.is_bounded()
        self.space = space
        self.n_bins = n_bins

        # initialize bins
        assert n_bins > 0, "Discretizer requires n_bins > 0"
        n_elements = 1
        tol = 1e-8
        self.dim = len(self.space.low)
        n_elements = n_bins ** self.dim
        self._bins = []
        self._open_bins = []
        for dd in range(self.dim):
            range_dd = self.space.high[dd] - self.space.low[dd]
            epsilon = range_dd / n_bins
            bins_dd = []
            for bb in range(n_bins + 1):
                val = self.space.low[dd] + epsilon * bb
                bins_dd.append(val)
            self._open_bins.append(tuple(bins_dd[1:]))
            bins_dd[-1] += tol  # "close" the last interval
            self._bins.append(tuple(bins_dd))

        # set observation space
        self.discrete_space = Discrete(n_elements)

        # List of discretized elements
        self.discretized_elements = np.zeros((self.dim, n_elements))
        for ii in range(n_elements):
            self.discretized_elements[:, ii] = self.get_coordinates(ii, False)

    def discretize(self, coordinates):
        return binary_search_nd(coordinates, self._bins)

    def get_coordinates(self, flat_index, randomize=False):
        assert self.discrete_space.contains(flat_index), "invalid flat_index"
        # get multi-index
        index = unravel_index_uniform_bin(flat_index, self.dim, self.n_bins)

        # get coordinates
        coordinates = np.zeros(self.dim)
        for dd in range(self.dim):
            coordinates[dd] = self._bins[dd][index[dd]]
            if randomize:
                range_dd = self.space.high[dd] - self.space.low[dd]
                epsilon = range_dd / self.n_bins
                coordinates[dd] += epsilon * self.space.rng.uniform()
        return coordinates
