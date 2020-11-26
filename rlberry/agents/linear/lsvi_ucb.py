import logging
import numpy as np
from rlberry.agents import Agent
from rlberry.spaces import Discrete


class LSVIUCBAgent(Agent):
    """
    A version of Least-Squares Value Iteration with UCB (LSVI-UCB),
    proposed by Jin et al. (2020).

    If bonus_scale_factor is 0.0, performs random exploration.

    TODO: Optimize using jit, improve logging.

    References
    ----------
    Jin, C., Yang, Z., Wang, Z., & Jordan, M. I. (2020, July).
    Provably efficient reinforcement learning with linear
    function approximation. In Conference on Learning Theory (pp. 2137-2143).
    """

    name = 'LSVI-UCB'
    fit_info = ("n_episodes", "episode_rewards")

    def __init__(self,
                 env,
                 horizon,
                 feature_map_fn,
                 n_episodes=100,
                 gamma=0.99,
                 bonus_scale_factor=1.0,
                 reg_factor=0.01,
                 verbose=1,
                 **kwargs):
        """
        Parameters
        ----------
        env : Model
            Online model of an environment.
        horizon : int
            Maximum length of each episode.
        feature_map_fn : function
            Function that returns a feature map instance
            (rlberry.agents.features.FeatureMap class).
        n_episodes : int
            number of episodes
        gamma : double
            Discount factor.
        bonus_scale_factor : double
            Constant by which to multiply the exploration bonus.
        reg_factor : double
            Linear regression regularization factor.
        verbose : int
            Verbosity level.
        """
        Agent.__init__(self, env, **kwargs)

        self.horizon = horizon
        self.feature_map = feature_map_fn()
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.bonus_scale_factor = bonus_scale_factor
        self.reg_factor = reg_factor
        self.verbose = verbose

        #
        if self.bonus_scale_factor == 0.0:
            self.name = 'LSVI-Random-Expl'

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf:
            logging.warning("{}: Reward range is infinity. ".format(self.name)
                            + "Clipping it to 1.")
            r_range = 1.0

        if self.gamma == 1.0:
            self.v_max = r_range * horizon
        else:
            self.v_max = r_range * (1.0 - np.power(self.gamma, self.horizon))\
                 / (1.0 - self.gamma)

        #
        assert isinstance(self.env.action_space, Discrete), \
            "LSVI-UCB requires discrete actions."

        #
        assert len(self.feature_map.shape) == 1
        self.dim = self.feature_map.shape[0]

        # attributes initialized in reset()
        self.episode = None
        self.lambda_mat = None      # lambda matrix
        self.lambda_mat_inv = None  # inverse of lambda matrix
        self.w_vec = None           # vector representation of Q
        self.w_policy = None        # representation of Q for final policy
        self.reward_hist = None     # reward history
        self.state_hist = None      # state history
        self.action_hist = None     # action history
        self.nstate_hist = None     # next state history
        #

        # aux variables (init in reset() too)
        self._rewards = None

        #
        self.reset()

    def reset(self, **kwargs):
        self.episode = 0
        self.lambda_mat = self.reg_factor * np.eye(self.dim)
        self.lambda_mat_inv = (1.0/self.reg_factor) * np.eye(self.dim)
        self.w_vec = np.zeros(self.dim)
        self.reward_hist = []
        self.state_hist = []
        self.action_hist = []
        self.nstate_hist = []
        #
        self._rewards = np.zeros(self.n_episodes)
        #
        self.w_policy = None

    def fit(self, **kwargs):
        info = {}
        for _ in range(self.n_episodes):
            self.run_episode()

            # log
            if self.verbose > 0:
                print(self._info_to_print())

        self.w_policy = self._run_lsvi(bonus_factor=0.0)

        info['n_episodes'] = self.n_episodes
        info['episode_rewards'] = self._rewards
        return info

    def policy(self, observation, **kwargs):
        q_w = self.w_policy
        assert q_w is not None
        #
        q_vec = self._compute_q_vec(q_w, observation, 0.0)
        return q_vec.argmax()

    def _optimistic_policy(self, observation):
        q_w = self.w_vec
        q_vec = self._compute_q_vec(q_w, observation, self.bonus_scale_factor)
        return q_vec.argmax()

    def run_episode(self):
        state = self.env.reset()
        episode_rewards = 0
        for hh in range(self.horizon):
            if self.bonus_scale_factor == 0.0:
                action = self.env.action_space.sample()
            else:
                action = self._optimistic_policy(state)
            next_state, reward, is_terminal, _ = self.env.step(action)

            feat = self.feature_map.map(state, action)
            outer_prod = np.outer(feat, feat)
            inv = self.lambda_mat_inv

            #
            self.lambda_mat += np.outer(feat, feat)
            # update inverse
            self.lambda_mat_inv -= \
                (inv @ outer_prod @ inv) / (1 + feat @ inv.T @ feat)

            # update history
            self.reward_hist.append(reward)
            self.state_hist.append(state)
            self.action_hist.append(action)
            self.nstate_hist.append(next_state)
            episode_rewards += reward

            #
            state = next_state
            if is_terminal:
                state = self.env.reset()
                break

        # update Q function representation
        self.w_vec = self._run_lsvi(self.bonus_scale_factor)

        # store data
        self._rewards[self.episode] = episode_rewards

        # update ep
        self.episode += 1

        return episode_rewards

    def _compute_q(self, q_w, state, action, bonus_factor):
        """q_w is the vector representation of the Q function."""
        feat = self.feature_map.map(state, action)
        inverse_counts = feat @ (self.lambda_mat_inv.T @ feat)
        bonus = bonus_factor * np.sqrt(inverse_counts)

        q = feat.dot(q_w) + bonus
        q = min(q, self.v_max)
        return q

    def _compute_q_vec(self, q_w, state, bonus_factor):
        A = self.env.action_space.n
        q_vec = np.zeros(A)
        for aa in range(A):
            q_vec[aa] = self._compute_q(q_w, state, aa, bonus_factor)
        return q_vec

    def _compute_targets(self, q_w, bonus_factor):
        T = len(self.reward_hist)
        b = np.zeros(self.dim)
        for tt in range(T):
            q_ns = self._compute_q_vec(q_w,
                                       self.nstate_hist[tt],
                                       bonus_factor)
            target = self.reward_hist[tt] + self.gamma*q_ns.max()
            feat = self.feature_map.map(self.state_hist[tt],
                                        self.action_hist[tt])
            b = b + target*feat
        return b

    def _run_lsvi(self, bonus_factor):
        # run value iteration
        q_w = np.zeros(self.dim)
        for hh in range(self.horizon - 1, -1, -1):
            # solve M x = b, where x = q_w, and M = self.lambda_mat
            b = self._compute_targets(q_w, bonus_factor)
            q_w = self.lambda_mat_inv.T @ b
        return q_w

    #
    # Logging
    #
    def _info_to_print(self):
        episode = self.episode
        avg_over = 10
        reward_per_ep = \
            self._rewards[max(0, episode-avg_over):episode + 1].mean()
        to_print = "[{}] episode = {}/{} ".format(self.name, episode+1,
                                                  self.n_episodes) \
            + "| reward/ep = {:0.2f} ".format(reward_per_ep)
        return to_print
