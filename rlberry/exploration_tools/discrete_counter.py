import numpy as np
from rlberry.exploration_tools.uncertainty_estimator \
    import UncertaintyEstimator
from rlberry.spaces import Discrete
from rlberry.utils.space_discretizer import Discretizer


class DiscreteCounter(UncertaintyEstimator):
    def __init__(self, observation_space, action_space, n_bins_obs=10,
                 n_bins_actions=10, **kwargs):
        UncertaintyEstimator.__init__(self, observation_space, action_space)

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

    def update(self, state, action, next_state, reward, **kwargs):
        state, action = self._preprocess(state, action)
        self.N_sa[state, action] += 1

    def measure(self, state, action, **kwargs):
        state, action = self._preprocess(state, action)
        n = np.maximum(1.0, self.N_sa[state, action])
        return 1.0/np.sqrt(n)

    def count(self, state, action):
        state, action = self._preprocess(state, action)
        return self.N_sa[state, action]
