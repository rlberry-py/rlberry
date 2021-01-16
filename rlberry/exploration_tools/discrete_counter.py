import numpy as np
from rlberry.exploration_tools.uncertainty_estimator \
    import UncertaintyEstimator
from rlberry.spaces import Discrete
from rlberry.utils.space_discretizer import Discretizer


class DiscreteCounter(UncertaintyEstimator):
    """
    Parameters
    ----------
    observation_space : spaces.Box or spaces.Discrete
    action_space : spaces.Box or spaces.Discrete
    n_bins_obs: int
        number of bins to discretize observation space
    n_bins_actions: int
        number of bins to discretize action space
    fast_rate : bool
        If true, returns bonuses in 1/n instead of 1/sqrt(n).
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 n_bins_obs=10,
                 n_bins_actions=10,
                 fast_rate=False,
                 **kwargs):
        UncertaintyEstimator.__init__(self, observation_space, action_space)

        self.fast_rate = fast_rate

        self.continuous_state = False
        self.continuous_action = False

        if isinstance(observation_space, Discrete):
            self.n_states = observation_space.n
        else:
            self.continuous_state = True
            self.state_discretizer = Discretizer(self.observation_space,
                                                 n_bins_obs)
            self.n_states = self.state_discretizer.discrete_space.n

        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            self.continuous_action = True
            self.action_discretizer = Discretizer(self.action_space,
                                                  n_bins_actions)
            self.n_actions = self.action_discretizer.discrete_space.n

        self.N_sa = np.zeros((self.n_states, self.n_actions))

    def _preprocess(self, state, action):
        if self.continuous_state:
            state = self.state_discretizer.discretize(state)
        if self.continuous_action:
            action = self.action_discretizer.discretize(action)
        return state, action

    def reset(self):
        self.N_sa = np.zeros((self.n_states, self.n_actions))

    def update(self, state, action, next_state=None, reward=None, **kwargs):
        state, action = self._preprocess(state, action)
        self.N_sa[state, action] += 1

    def measure(self, state, action, **kwargs):
        state, action = self._preprocess(state, action)
        n = np.maximum(1.0, self.N_sa[state, action])
        if self.fast_rate:
            return 1.0/n
        return 1.0/np.sqrt(n)

    def count(self, state, action):
        state, action = self._preprocess(state, action)
        return self.N_sa[state, action]

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
