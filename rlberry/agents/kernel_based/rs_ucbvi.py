import logging
import numpy as np

import gym.spaces as spaces
from rlberry.agents import IncrementalAgent
from rlberry.agents.dynprog.utils import backward_induction
from rlberry.agents.dynprog.utils import backward_induction_in_place
from rlberry.agents.kernel_based.common import map_to_representative
from rlberry.utils.writers import PeriodicWriter


logger = logging.getLogger(__name__)


class RSUCBVIAgent(IncrementalAgent):
    """
    Value iteration with exploration bonuses for continuous-state environments,
    using a online discretization strategy:
    - Build (online) a set of representative states
    - Estimate transtions an rewards on the finite set of representative states
    and actions.

    Criterion: finite-horizon with discount factor gamma.
    If the discount is not 1, only the Q function at h=0 is used.

    The recommended policy after all the episodes is computed without
    exploration bonuses.

    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
    n_episodes : int
        number of episodes
    gamma : double
        Discount factor in [0, 1]. If gamma is 1.0, the problem is set to
        be finite-horizon.
    horizon : int
        Horizon of the objective function. If None and gamma<1, set to
        1/(1-gamma).
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
    bonus_scale_factor : double
        Constant by which to multiply the exploration bonus, controls
        the level of exploration.
    bonus_type : string
        Type of exploration bonus. Currently, only "simplified_bernstein"
        is implemented. If `reward_free` is true, this parameter is ignored
        and the algorithm uses 1/n bonuses.
    reward_free : bool
        If true, ignores rewards and uses only 1/n bonuses.

    References
    ----------
    .. [1] Azar, Mohammad Gheshlaghi, Ian Osband, and Rémi Munos.
    "Minimax regret bounds for reinforcement learning."
    Proceedings of the 34th ICML, 2017.

    .. [2] Strehl, Alexander L., and Michael L. Littman.
    "An analysis of model-based interval estimation for Markov decision
    processes."
     Journal of Computer and System Sciences 74.8 (2008): 1309-1331.

    .. [3] Kveton, Branislav, and Georgios Theocharous.
    "Kernel-Based Reinforcement Learning on Representative States."
    AAAI, 2012.

    .. [4] Domingues, O. D., Ménard, P., Pirotta, M., Kaufmann, E., & Valko, M.(2020).
    A kernel-based approach to non-stationary reinforcement learning in metric
    spaces.
    arXiv preprint arXiv:2007.05078.
    """

    name = "RSUCBVI"
    fit_info = ("n_episodes", "episode_rewards")

    def __init__(self, env,
                 n_episodes=1000,
                 gamma=0.99,
                 horizon=100,
                 lp_metric=2,
                 scaling=None,
                 min_dist=0.1,
                 max_repr=1000,
                 bonus_scale_factor=1.0,
                 bonus_type="simplified_bernstein",
                 reward_free=False,
                 **kwargs):
        # init base class
        IncrementalAgent.__init__(self, env, **kwargs)

        self.n_episodes = n_episodes
        self.gamma = gamma
        self.horizon = horizon
        self.lp_metric = lp_metric
        self.min_dist = min_dist
        self.bonus_scale_factor = bonus_scale_factor
        self.bonus_type = bonus_type
        self.reward_free = reward_free

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # other checks
        assert gamma >= 0 and gamma <= 1.0
        if self.horizon is None:
            assert gamma < 1.0, \
                "If no horizon is given, gamma must be smaller than 1."
            self.horizon = int(np.ceil(1.0 / (1.0 - gamma)))

        # state dimension
        self.state_dim = self.env.observation_space.shape[0]

        # compute scaling, if it is None
        if scaling is None:
            # if high and low are bounded
            if (self.env.observation_space.high == np.inf).sum() == 0 \
                    and (self.env.observation_space.low == -np.inf).sum() == 0:
                scaling = self.env.observation_space.high \
                    - self.env.observation_space.low
                # if high or low are unbounded
            else:
                scaling = np.ones(self.state_dim)
        else:
            assert scaling.ndim == 1
            assert scaling.shape[0] == self.state_dim
        self.scaling = scaling

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf or r_range == 0.0:
            logger.warning("{}: Reward range is  zero or infinity. ".format(self.name)
                           + "Setting it to 1.")
            r_range = 1.0

        if self.gamma == 1.0:
            self.v_max = r_range * horizon
        else:
            self.v_max = r_range * (1.0 - np.power(self.gamma, self.horizon)) \
                                                        / (1.0 - self.gamma)

        # number of representative states and number of actions
        if max_repr is None:
            max_repr = int(np.ceil((1.0 * np.sqrt(self.state_dim) /
                                    self.min_dist) ** self.state_dim))
        self.max_repr = max_repr

        # current number of representative states
        self.M = None
        self.A = self.env.action_space.n

        # declaring variables
        self.episode = None  # current episode
        self.representative_states = None  # coordinates of all repr states
        self.N_sa = None  # visits to (s, a)
        self.N_sas = None  # visits to (s, a, s')
        self.S_sa = None  # sum of rewards at (s, a)
        self.B_sa = None  # bonus at (s, a)
        self.Q = None  # Q function
        self.V = None  # V function

        self.Q_policy = None  # Q function for recommended policy

        # initialize
        self.reset()

    def reset(self, **kwargs):
        self.M = 0
        self.representative_states = np.zeros((self.max_repr, self.state_dim))
        self.N_sa = np.zeros((self.max_repr, self.A))
        self.N_sas = np.zeros((self.max_repr, self.A, self.max_repr))
        self.S_sa = np.zeros((self.max_repr, self.A))
        self.B_sa = self.v_max * np.ones((self.max_repr, self.A))

        self.R_hat = np.zeros((self.max_repr, self.A))
        self.P_hat = np.zeros((self.max_repr, self.A, self.max_repr))

        self.V = np.zeros((self.horizon, self.max_repr))
        self.Q = np.zeros((self.horizon, self.max_repr, self.A))
        self.Q_policy = None

        self.episode = 0

        # info
        self._rewards = np.zeros(self.n_episodes)
        self._cumul_rewards = np.zeros(self.n_episodes)

        # default writer
        self.writer = PeriodicWriter(self.name,
                                     log_every=5*logger.getEffectiveLevel())

    def policy(self, state, hh=0, **kwargs):
        assert self.Q_policy is not None
        repr_state = self._map_to_repr(state, False)

        # no discount
        if self.gamma == 1.0:
            return self.Q_policy[hh, repr_state, :].argmax()
        # discounted
        return self.Q_policy[0, repr_state, :].argmax()

    def partial_fit(self, fraction, **kwargs):
        assert 0.0 < fraction <= 1.0
        n_episodes_to_run = int(np.ceil(fraction*self.n_episodes))
        count = 0
        while count < n_episodes_to_run and self.episode < self.n_episodes:
            self._run_episode()
            count += 1

        # compute Q function for the recommended policy
        self.Q_policy, _ = backward_induction(self.R_hat[:self.M, :],
                                              self.P_hat[:self.M, :, :self.M],
                                              self.horizon, self.gamma)

        info = {"n_episodes": self.episode,
                "episode_rewards": self._rewards[:self.episode]}
        return info

    def _map_to_repr(self, state, accept_new_repr=True):
        repr_state = map_to_representative(state,
                                           self.lp_metric,
                                           self.representative_states,
                                           self.M,
                                           self.min_dist,
                                           self.scaling,
                                           accept_new_repr)
        # check if new representative state
        if repr_state == self.M:
            self.M += 1
        return repr_state

    def _update(self, state, action, next_state, reward):
        repr_state = self._map_to_repr(state)
        repr_next_state = self._map_to_repr(next_state)

        self.N_sa[repr_state, action] += 1
        self.N_sas[repr_state, action, repr_next_state] += 1
        self.S_sa[repr_state, action] += reward

        self.R_hat[repr_state, action] = self.S_sa[repr_state, action] \
            / self.N_sa[repr_state, action]
        self.P_hat[repr_state, action, :] = self.N_sas[repr_state, action, :] \
            / self.N_sa[repr_state, action]
        self.B_sa[repr_state, action] = \
            self._compute_bonus(self.N_sa[repr_state, action])

    def _compute_bonus(self, n):
        # reward-free
        if self.reward_free:
            bonus = 1.0 / n
            return bonus

        # not reward-free
        if self.bonus_type == "simplified_bernstein":
            bonus = self.bonus_scale_factor * np.sqrt(1.0 / n) + self.v_max / n
            bonus = min(bonus, self.v_max)
            return bonus
        else:
            raise NotImplementedError(
                "Error: bonus type {} not implemented".format(self.bonus_type))

    def _get_action(self, state, hh=0):
        assert self.Q is not None
        repr_state = self._map_to_repr(state, False)

        # no discount
        if self.gamma == 1.0:
            return self.Q[hh, repr_state, :].argmax()
        # discounted
        return self.Q[0, repr_state, :].argmax()

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        for hh in range(self.horizon):
            action = self._get_action(state, hh)
            next_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward  # used for logging only

            if self.reward_free:
                reward = 0.0  # set to zero before update if reward_free

            self._update(state, action, next_state, reward)

            state = next_state
            if done:
                break

        # run backward induction
        backward_induction_in_place(
            self.Q[:, :self.M, :], self.V[:, :self.M],
            self.R_hat[:self.M, :] + self.B_sa[:self.M, :],
            self.P_hat[:self.M, :, :self.M],
            self.horizon, self.gamma, self.v_max)

        ep = self.episode
        self._rewards[ep] = episode_rewards
        self._cumul_rewards[ep] = episode_rewards \
            + self._cumul_rewards[max(0, ep - 1)]

        self.episode += 1
        #
        if self.writer is not None:
            avg_reward = self._cumul_rewards[ep]/max(1, ep)

            self.writer.add_scalar("ep reward", episode_rewards, self.episode)
            self.writer.add_scalar("avg reward", avg_reward, self.episode)
            self.writer.add_scalar("representative states", self.M, self.episode)

        # return sum of rewards collected in the episode
        return episode_rewards
