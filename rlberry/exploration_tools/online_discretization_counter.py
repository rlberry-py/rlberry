import logging
import numpy as np
from rlberry.utils.jit_setup import numba_jit
from rlberry.exploration_tools.uncertainty_estimator \
    import UncertaintyEstimator
from gym.spaces import Box, Discrete
from rlberry.utils.metrics import metric_lp

logger = logging.getLogger(__name__)


@numba_jit
def map_to_representative(state,
                          lp_metric,
                          representative_states,
                          n_representatives,
                          min_dist,
                          scaling,
                          accept_new_repr):
    """
    Map state to representative state.
    """
    dist_to_closest = np.inf
    argmin = -1
    for ii in range(n_representatives):
        dist = metric_lp(state, representative_states[ii, :],
                         lp_metric, scaling)
        if dist < dist_to_closest:
            dist_to_closest = dist
            argmin = ii

    max_representatives = representative_states.shape[0]
    if dist_to_closest > min_dist \
        and n_representatives < max_representatives \
            and accept_new_repr:
        new_index = n_representatives
        representative_states[new_index, :] = state
        return new_index, 0.0
    return argmin, dist_to_closest


class OnlineDiscretizationCounter(UncertaintyEstimator):
    """
    Note: currently, only implemented for continuous (Box) states and
    discrete actions.

    Parameters
    ----------
    observation_space : spaces.Box
    action_space : spaces.Discrete
    lp_metric: int
        The metric on the state space is the one induced by the p-norm,
        where p = lp_metric. Default = 2, for the Euclidean metric.
    scaling: numpy.ndarray
        Must have the same size as state array, used to scale the states
        before computing the metric.
        If None, set to:
        - (env.observation_space.high - env.observation_space.low) if high
        and low are bounded
        - np.ones(env.observation_space.shape[0]) if high or low are
        unbounded
    min_dist: double
        Minimum distance between two representative states
    max_repr: int
        Maximum number of representative states.
        If None, it is set to  (sqrt(d)/min_dist)**d, where d
        is the dimension of the state space
    fast_rate : bool
        If true, returns bonuses in 1/n instead of 1/sqrt(n).
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 lp_metric=2,
                 min_dist=0.1,
                 max_repr=1000,
                 scaling=None,
                 fast_rate=False,
                 **kwargs):
        UncertaintyEstimator.__init__(self, observation_space, action_space)

        assert isinstance(action_space, Discrete)
        assert isinstance(observation_space, Box)

        self.lp_metric = lp_metric
        self.min_dist = min_dist
        self.max_repr = max_repr
        self.state_dim = self.observation_space.shape[0]
        self.n_actions = self.action_space.n
        self.fast_rate = fast_rate

        # compute scaling, if it is None
        if scaling is None:
            # if high and low are bounded
            if self.observation_space.is_bounded():
                scaling = self.observation_space.high \
                    - self.observation_space.low
                # if high or low are unbounded
            else:
                scaling = np.ones(self.state_dim)
        else:
            assert scaling.ndim == 1
            assert scaling.shape[0] == self.state_dim
        self.scaling = scaling

        # initialize
        self.n_representatives = None
        self.representative_states = None
        self.N_sa = None
        self.reset()

    def reset(self):
        self.n_representatives = 0
        self.representative_states = np.zeros((self.max_repr, self.state_dim))
        self.N_sa = np.zeros((self.max_repr, self.n_actions))

        self._overflow_warning = False

    def _get_representative_state(self, state, accept_new_repr=True):
        state_idx, dist_to_closest \
            = map_to_representative(state,
                                    self.lp_metric,
                                    self.representative_states,
                                    self.n_representatives,
                                    self.min_dist,
                                    self.scaling,
                                    accept_new_repr)
        # check if new representative state
        if state_idx == self.n_representatives:
            self.n_representatives += 1

        if self.n_representatives >= self.max_repr \
                and (not self._overflow_warning):
            logger.warning("OnlineDiscretizationCounter reached \
the maximum number of representative states.")
            self._overflow_warning = True

        return state_idx, dist_to_closest

    def update(self, state, action, next_state, reward, **kwargs):
        state_idx, _ = self._get_representative_state(state)
        self.N_sa[state_idx, action] += 1

    def measure(self, state, action, **kwargs):
        n = np.maximum(1.0, self.count(state, action))
        if self.fast_rate:
            return 1.0/n
        return 1.0/np.sqrt(n)

    def count(self, state, action):
        state_idx, dist_to_closest = self._get_representative_state(
                                            state,
                                            accept_new_repr=False)
        # if state is too far from the closest representative,
        # its count is zero.
        if dist_to_closest > self.min_dist:
            return 0.0
        return self.N_sa[state_idx, action]

    def get_n_visited_states(self):
        """
        Returns the number of different states sent to the .update() function.
        For continuous state spaces, counts the number of different discretized states.
        """
        n_visited_states = (self.N_sa.sum(axis=1) > 0).sum()
        return n_visited_states

    def get_entropy(self):
        """
        Returns the entropy of the empirical distribution over states, induced by the state counts.
        Uses log2.
        """
        visited = self.N_sa.sum(axis=1) > 0
        if visited.sum() == 0.0:
            return 0.0
        # number of visits of visited states only
        n_visits = self.N_sa[visited, :].sum(axis=1)
        # empirical distribution
        dist = n_visits/n_visits.sum()
        entropy = (-dist*np.log2(dist)).sum()
        return entropy
