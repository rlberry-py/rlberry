import logging
import numpy as np
from rlberry.agents import Agent
from gym.spaces import Discrete
from rlberry.utils.writers import PeriodicWriter
from rlberry.utils.jit_setup import numba_jit


logger = logging.getLogger(__name__)


class KOVIAgent(Agent):
    """
    A version of Kernelized Optimistic Least-Squares Value Iteration (KOVI),
    proposed by Jin et al. (2020).

    If bonus_scale_factor is 0.0, performs random exploration.

    Parameters
    ----------
    env : Model
        Online model of an environment.
    horizon : int
        Maximum length of each episode.
    feature_map_fn : function(env, kwargs)
        Function that returns a feature map instance
        (rlberry.agents.features.FeatureMap class).
    feature_map_kwargs:
        kwargs for feature_map_fn
    n_episodes : int
        number of episodes
    gamma : double
        Discount factor.
    bonus_scale_factor : double
        Constant by which to multiply the exploration bonus.
    reg_factor : double
        Linear regression regularization factor.

    References
    ----------

    """

    name = 'KOVI'

    def __init__(self,
                 env,
                 horizon,
                 pd_kernel_fn,
                 pd_kernel_kwargs=None,
                 n_episodes=100,
                 gamma=0.99,
                 bonus_scale_factor=1.0,
                 reg_factor=0.1,
                 **kwargs):
        Agent.__init__(self, env, **kwargs)

        self.horizon = horizon
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.bonus_scale_factor = bonus_scale_factor
        self.reg_factor = reg_factor
        pd_kernel_kwargs = pd_kernel_kwargs or {}
        self.pd_kernel = pd_kernel_fn

        #
        if self.bonus_scale_factor == 0.0:
            self.name = 'KOVI-Random-Expl'

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf:
            logger.warning("{}: Reward range is infinity. ".format(self.name)
                           + "Clipping it to 1.")
            r_range = 1.0

        if self.gamma == 1.0:
            self.v_max = r_range * horizon
        else:
            self.v_max = r_range * (1.0 - np.power(self.gamma, self.horizon))\
                 / (1.0 - self.gamma)

        #
        assert isinstance(self.env.action_space, Discrete), \
            "KOVI requires discrete actions."

        # attributes initialized in reset()
        self.episode = None
        self.gram_mat = None        # Gram matrix
        self.gram_mat_inv = None    # inverse of Gram matrix
        self.alphas = None          #

        self.reward_hist = None     # reward history
        self.state_hist = None      # state history
        self.action_hist = None     # action history
        self.nstate_hist = None     # next state history

        # aux variables (init in reset() too)
        self._rewards = None

        # default writer
        self.writer = PeriodicWriter(self.name,
                                     log_every=15)

        #
        self.reset()

    def reset(self, **kwargs):
        self.episode = 0
        self.total_time_steps = 0
        self.targets = [[] for _ in range(self.horizon)]
        self.gram_mat = [np.array([self.reg_factor]) for _ in range(self.horizon)]  # self.reg_factor * np.eye(self.dim)
        self.gram_mat_inv = [np.array([1.0 / self.reg_factor]) for _ in range(self.horizon)]
        self.alphas = [np.array([1]) for _ in range(self.horizon)]

        self.reward_hist = [[] for _ in range(self.horizon)]
        self.state_hist = [[] for _ in range(self.horizon + 1)]
        self.action_hist = [[] for _ in range(self.horizon)]
        self.nstate_hist = []

        # episode rewards
        self._rewards = np.zeros(self.n_episodes)

    def bonus(self, step, state, action):
        # b_h^t (s, a)
        kh = self.score_vector(step, state, action)

        tmp1 = self.pd_kernel((state, action))

        print('kh', kh)
        print('gram inv', self.gram_mat_inv[step])
        if self.episode == 0:
            tmp2 = kh * self.gram_mat_inv[step] * kh
        else:
            tmp2 = kh.T @ self.gram_mat_inv[step] @ kh

        bonus = np.sqrt((tmp1 - tmp2) / self.reg_factor)
        # bonus = np.sqrt((self.pd_kernel((state, action)) - kh.T @ self.gram_mat_inv[step] @ kh) / self.reg_factor)
        return bonus


    def score_vector(self, step, state, action):
        # k_h^t (s, a)
        previous_states = self.state_hist[step]
        previous_actions = self.action_hist[step]

        similarities = [self.pd_kernel((s, a), (state, action)) for s, a in zip(previous_states, previous_actions)]
        return np.array(similarities)

    def q_function(self, step, state, action):
        # Q_h^t (s, a)
        mean = self.score_vector(step, state, action).T @ self.alphas[step]
        bonus = self.bonus(step, state, action)
        Q = mean + self.bonus_scale_factor * bonus

        return max(min(Q, self.horizon - step + 1), 0)

    def value_function(self, step, state):
        # V_h^t (s)
        if isinstance(state, list):
            n_states = len(state)

            if step == self.horizon:
                return np.zeros(n_states)
            else:
                return np.array([np.max([self.q_function(step, s, a) for a in range(self.env.action_space.n)]) for s in state])

        else:
            if step == self.horizon:
                return 0
            else:
                return np.max([self.q_function(step, state, a) for a in range(self.env.action_space.n)])

    def policy(self, state, **kwargs):
        return

    def _optimistic_policy(self, step, state):
        return np.argmax([self.q_function(step, state, a) for a in range(self.env.action_space.n)])

    def fit(self, **kwargs):
        info = {}
        for ep in range(self.n_episodes):
            state = self.env.reset()

            print('Episode: {}'.format(ep))
            self.episode = ep
            # run episode
            print('Run episode --')
            for step in range(self.horizon):

                print('Step: {}'.format(step))

                if ep == 0:
                    action = 0
                else:
                    action = self._optimistic_policy(step, state)
                next_state, reward, is_terminal, _ = self.env.step(action)

                # update history
                self.reward_hist[step].append(reward)
                self.state_hist[step].append(state)
                self.action_hist[step].append(action)

                self._rewards[ep] += reward

                state = next_state

            # run kovi
            print('Run KOVI --')
            for step in range(self.horizon - 1, -1, -1):
                print('Step: {}'.format(step))

                # y_h^t
                self.targets[step] = self.reward_hist[step] + self.value_function(step + 1, self.state_hist[step])

                if ep > 0:
                    # K_h^t
                    Kaa = self.gram_mat[step]
                    Kab = self.score_vector(step, self.state_hist[step][-1], self.action_hist[step][-1])[:-1].reshape(-1, 1)
                    Kba = Kab.reshape(1, -1)
                    Kbb = np.array([self.pd_kernel((self.state_hist[step][-1], self.action_hist[step][-1]))])

                    tmp1 = np.hstack((Kaa, Kab))
                    tmp2 = np.hstack((Kba, Kbb))

                    self.gram_mat[step] = np.vstack((tmp1, tmp2))

                    # (K_h^t)^-1
                    Kdd = 1.0 / (Kbb + self.reg_factor - Kba @ self.gram_mat_inv[step] @ Kab)
                    Kcc = self.gram_mat_inv[step] + Kdd * self.gram_mat_inv[step] @ np.outer(Kab, Kab) @ self.gram_mat_inv[step]
                    Kcd = - Kdd * self.gram_mat_inv[step] @ Kab
                    Kdc = - Kdd * Kba @ self.gram_mat_inv[step]
                    self.gram_mat_inv[step] = np.vstack((np.hstack((Kcc, Kcd)), np.hstack((Kdc, Kdd))))

                # alpha_h^t
                self.alphas[step] = self.gram_mat_inv[step] @ self.targets[step]
                if ep == 0:
                    self.alphas[step] = np.array([self.alphas[step]])


        info['n_episodes'] = self.n_episodes
        info['episode_rewards'] = self._rewards
        return info




