import numpy as np
import logging

import rlberry.spaces as spaces
from rlberry.envs.interface import Model

logger = logging.getLogger(__name__)


class FiniteMDP(Model):
    """
    Base class for a finite MDP.

    Terminal states are set to be absorbing, and
    are determined by the is_terminal() method,
    which can be overriden (and returns false by default).

    Parameters
    ----------
    R : numpy.ndarray
    P : numpy.ndarray
    initial_state_distribution : numpy.ndarray or int
        array of size (S,) containing the initial state distribution
        or an integer representing the initial/default state

    Attributes
    ----------
    R : numpy.ndarray
        array of shape (S, A) containing the mean rewards, where
        S = number of states;  A = number of actions.
    P : numpy.ndarray
        array of shape (S, A, S) containing the transition probabilities,
        where P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a).
    """

    def __init__(self, R, P, initial_state_distribution=0):
        Model.__init__(self)
        self.initial_state_distribution = initial_state_distribution
        S, A = R.shape

        self.S = S
        self.A = A

        self.R = R
        self.P = P

        self.observation_space = spaces.Discrete(S)
        self.action_space = spaces.Discrete(A)
        self.reward_range = (self.R.min(), self.R.max())

        self.state = None

        self._states = np.arange(S)
        self._actions = np.arange(A)

        self.reset()
        self._process_terminal_states()
        self._check()

    def reset(self):
        """
        Reset the environment to a default state.
        """
        if isinstance(self.initial_state_distribution, np.ndarray):
            self.state = self.rng.choice(self._states,
                                         p=self.initial_state_distribution)
        else:
            self.state = self.initial_state_distribution
        return self.state

    def _process_terminal_states(self):
        """
        Adapt transition array P so that terminal states
        are absorbing.
        """
        for ss in range(self.S):
            if self.is_terminal(ss):
                self.P[ss, :, :] = 0.0
                self.P[ss, :, ss] = 1.0

    def _check(self):
        """
        Check consistency of the MDP
        """
        # Check initial_state_distribution
        if isinstance(self.initial_state_distribution, np.ndarray):
            assert abs(self.initial_state_distribution.sum() - 1.0) < 1e-15
        else:
            assert self.initial_state_distribution >= 0
            assert self.initial_state_distribution < self.S

        # Check that P[s,a, :] is a probability distribution
        for s in self._states:
            for a in self._actions:
                assert abs(self.P[s, a, :].sum() - 1.0) < 1e-15

        # Check that dimensions match
        S1, A1 = self.R.shape
        S2, A2, S3 = self.P.shape
        assert S1 == S2 == S3
        assert A1 == A2

    def sample(self, state, action):
        """
        Sample a transition s' from P(s'|state, action).
        """
        prob = self.P[state, action, :]
        next_state = self.rng.choice(self._states, p=prob)
        reward = self.reward_fn(state, action, next_state)
        done = self.is_terminal(self.state)
        info = {}
        return next_state, reward, done, info

    def step(self, action):
        assert action in self._actions, "Invalid action!"
        next_state, reward, done, info = self.sample(self.state, action)
        self.state = next_state
        return next_state, reward, done, info

    def is_terminal(self, state):
        """
        Returns true if a state is terminal.
        """
        return False

    def reward_fn(self, state, action, next_state):
        """
        Reward function. Returns mean reward at (state, action) by default.

        Parameters
        ----------
        state : int
            current state
        action : int
            current action
        next_state :
            next state

        Returns:
            reward : float
        """
        return self.R[state, action]

    def log(self):
        """
        Print the structure of the MDP.
        """
        indent = '    '
        for s in self._states:
            logger.info(f"State {s} {indent}")
            for a in self._actions:
                logger.info(f"{indent} Action {a}")
                for ss in self._states:
                    if self.P[s, a, ss] > 0.0:
                        logger.info(f'{2 * indent} transition to {ss} '
                                    f'with prob {self.P[s, a, ss]: .2f}')
            logger.info("~~~~~~~~~~~~~~~~~~~~")


# if __name__ == '__main__':
#     S = 3
#     A = 2

#     R = np.random.uniform(0, 1, (S, A))
#     P = np.random.uniform(0, 1, (S, A, S))
#     initial_state_distr = 1  # np.ones(S)/S
#     for ss in range(S):
#         for aa in range(A):
#             P[ss, aa, :] /= P[ss, aa, :].sum()

#     env = FiniteMDP(R, P, initial_state_distribution=initial_state_distr)
