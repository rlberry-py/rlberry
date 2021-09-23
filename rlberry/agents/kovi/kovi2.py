import logging
import numpy as np
from rlberry.agents import Agent
from gym.spaces import Discrete
from rlberry.utils.writers import PeriodicWriter
from rlberry.utils.jit_setup import numba_jit

# from typing import List, Callable, Union
from numba.typed import List
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

logger = logging.getLogger(__name__)


class KOVIAgent(Agent):
    """
    A version of Least-Squares Value Iteration with UCB (LSVI-UCB),
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
    Jin, C., Yang, Z., Wang, Z., & Jordan, M. I. (2020, July).
    Provably efficient reinforcement learning with linear
    function approximation. In Conference on Learning Theory (pp. 2137-2143).
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

        self.use_jit = True
        self.horizon = horizon
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.bonus_scale_factor = bonus_scale_factor
        self.reg_factor = reg_factor
        self.total_time_steps = 0

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
        self.alphas = None          # vector representations of Q
        self.reward_hist = None     # reward history
        self.state_hist = None      # state history
        self.action_hist = None     # action history
        self.nstate_hist = None     # next state history
        self.rkhs_norm_hist = None  # norm history

        self.feat_hist = None             # feature history
        self.feat_ns_all_actions = None   # next state features for all actions

        self.new_gram_mat = None
        self.new_gram_mat_inv = None
        #

        # aux variables (init in reset() too)
        self._rewards = None

        # default writer
        self.writer = PeriodicWriter(self.name, log_every=15)
        # 5*logger.getEffectiveLevel()

        #
        self.reset()

    def reset(self, **kwargs):
        self.episode = 0
        self.total_time_steps = 0
        self.gram_mat = [np.array([self.reg_factor]).reshape(1, 1) for _ in range(self.horizon)]
        self.gram_mat_inv = [np.array([(1.0 / self.reg_factor)]).reshape(1, 1) for _ in range(self.horizon)]

        self.new_gram_mat = [np.array([self.reg_factor]).reshape(1, 1) for _ in range(self.horizon)]
        self.new_gram_mat_inv = [np.array([(1.0 / self.reg_factor)]).reshape(1, 1) for _ in range(self.horizon)]

        self.reward_hist = np.zeros(self.n_episodes * self.horizon)
        self.state_hist = []
        self.action_hist = []
        self.nstate_hist = []
        self.rkhs_norm_hist = []

        self.hist_reward = [[] for _ in range(self.horizon)]
        self.hist_rkhs_norm = np.zeros(
            (self.horizon, self.n_episodes, self.env.action_space.n)
        )
        self.hist_nstate = [[] for _ in range(self.horizon)]
        self.nstate_feat_hist = np.zeros(
            (self.horizon, self.n_episodes, self.n_episodes, self.env.action_space.n)
        )

        # episode rewards
        self._rewards = np.zeros(self.n_episodes)

        #
        self.feat_hist = [[] for _ in range(self.n_episodes * self.horizon)]
        self.feat_ns_all_actions = [
            [[] for _ in range(self.env.action_space.n)] for _ in range(self.n_episodes * self.horizon)
        ]

        #
        self.alphas = [np.array([0.]) for _ in range(self.horizon)]

    def fit(self, **kwargs):
        info = {}
        for ep in range(self.n_episodes):
            self.run_episode()
            if self.bonus_scale_factor > 0.0 or ep == self.n_episodes - 1:
                # update Q function representation

                for step in range(self.horizon):
                    self.gram_mat[step] = self.new_gram_mat[step]
                    self.gram_mat_inv[step] = self.new_gram_mat_inv[step]

                if self.use_jit:
                    self.alphas = self._run_kovi(self.bonus_scale_factor)
                else:
                    self._run_kovi(self.bonus_scale_factor)

        info['n_episodes'] = self.n_episodes
        info['episode_rewards'] = self._rewards
        return info

    def run_episode(self):
        # print(f'Episode: {self.episode}')

        state = self.env.reset()
        episode_rewards = 0

        self.new_gram_mat = np.zeros((self.horizon, self.episode + 1, self.episode + 1))
        self.new_gram_mat_inv = np.zeros((self.horizon, self.episode + 1, self.episode + 1))

        for step in range(self.horizon):
            # print(f'Step {step}')
            if self.bonus_scale_factor == 0.0 or self.episode == 0:
                action = self.env.action_space.sample()
            else:
                action = self._optimistic_policy(step, state)

            next_state, reward, is_terminal, _ = self.env.step(action)
            # if is_terminal:
            #     reward = 0.

            # update Gram matrix and its inverse
            if self.episode == 0:
                self.new_gram_mat[step] += self.pd_kernel(state, action)
                self.new_gram_mat_inv[step] = 1. / self.new_gram_mat[step]

            elif self.episode >= 1:
                feat = self._score_vector(step, state, action)
                outer_prod = np.outer(feat, feat)

                # update Gram matrix
                self.new_gram_mat[step, :-1, :-1] = self.gram_mat[step]
                self.new_gram_mat[step, -1, :-1] = feat
                self.new_gram_mat[step, :-1, -1] = feat
                self.new_gram_mat[step, -1, -1] = self.pd_kernel(state, action) + self.reg_factor

                # update Gram inverse
                K22 = 1.0 / (self.gram_mat[step][-1, -1] + self.reg_factor - feat.T @ self.gram_mat_inv[step] @ feat)
                K11 = self.gram_mat_inv[step] + K22 * self.gram_mat_inv[step] @ outer_prod @ self.gram_mat_inv[step]
                K12 = - K22 * self.gram_mat_inv[step] @ feat
                K21 = - K22 * feat.T @ self.gram_mat_inv[step]

                self.new_gram_mat_inv[step, :-1, :-1] = K11
                self.new_gram_mat_inv[step, :-1, -1] = K12
                self.new_gram_mat_inv[step, -1, :-1] = K21
                self.new_gram_mat_inv[step, -1, -1] = K22

                # store next features
                # tt = self.total_time_steps
                # self.feat_hist[tt] = feat
                # for aa in range(self.env.action_space.n):
                #     self.feat_ns_all_actions[tt][aa] = self._score_vector(step + 1, next_state, aa)

                for tau in range(self.episode):
                    pns = self.hist_nstate[step][tau]
                    for a in range(self.env.action_space.n):
                        similarity = self.pd_kernel(next_state, action, pns, a)
                        self.nstate_feat_hist[step, tau, self.episode, a] = similarity

            # update history
            self.reward_hist[self.total_time_steps] = reward
            self.state_hist.append(state)
            self.action_hist.append(action)
            self.nstate_hist.append(next_state)
            self.rkhs_norm_hist.append(self.pd_kernel(state, action))

            for a in range(self.env.action_space.n):
                self.hist_rkhs_norm[step, self.episode, a] = self.pd_kernel(next_state, a)

            self.hist_nstate[step].append(next_state)
            self.hist_reward[step].append(reward)

            # increments
            self.total_time_steps += 1
            episode_rewards += reward

            # update current state
            state = next_state

            # if is_terminal:
            #     break

        # store data
        self._rewards[self.episode] = episode_rewards

        # update episode
        self.episode += 1

        # log data
        if self.writer is not None:
            self.writer.add_scalar("episode", self.episode, None)
            self.writer.add_scalar("ep reward", episode_rewards)

        return episode_rewards

    def policy(self, observation, **kwargs):
        # return self._compute_q_vec(self.alphas[0], 0, observation, self.bonus_scale_factor).argmax()
        return self._compute_q_vec(0, observation, 0.0).argmax()

    def _optimistic_policy(self, step, observation):
        # return self._compute_q_vec(self.alphas[step], step, observation, self.bonus_scale_factor).argmax()
        return self._compute_q_vec(step, observation, self.bonus_scale_factor).argmax()

    def _compute_q_vec(self, step, state, bonus_factor):
        n_actions = self.env.action_space.n
        q_vec = np.zeros(n_actions)

        if step == self.horizon:
            return q_vec

        else:
            for aa in range(n_actions):
                feat = self._score_vector(step, state, aa)

                inverse_counts = self.pd_kernel(state, aa) - feat @ self.gram_mat_inv[step] @ feat
                bonus = bonus_factor * np.sqrt(inverse_counts) + self.v_max * inverse_counts * (bonus_factor > 0.0)

                q_vec[aa] = feat.T @ self.alphas[step] + bonus
                q_vec[aa] = min(q_vec[aa], self.v_max)

            return q_vec

    def _score_vector(self, step, state, action):
        # k_h^t (s, a)
        prev_states = self._get_previous_states(step, by='episode')
        prev_actions = self._get_previous_actions(step, by='episode')

        similarities = [self.pd_kernel(s, a, state2=state, action2=action) for s, a in zip(prev_states, prev_actions)]
        return np.array(similarities)

    def _get_previous_states(self, index, by='episode'):
        if by == 'episode':
            return [self.state_hist[index + ep * self.horizon] for ep in range(self.episode)]
        elif by == 'step':
            return [self.state_hist[step + index * self.horizon] for step in range(self.horizon)]

    def _get_previous_actions(self, index, by='episode'):
        if by == 'episode':
            return [self.action_hist[index + ep * self.horizon] for ep in range(self.episode)]
        elif by == 'step':
            return [self.action_hist[step + index * self.horizon] for step in range(self.horizon)]

    def _get_previous_rewards(self, index, by='episode'):
        if by == 'episode':
            return [self.reward_hist[index + ep * self.horizon] for ep in range(self.episode)]
        elif by == 'step':
            return [self.reward_hist[step + index * self.horizon] for step in range(self.horizon)]

    def _run_kovi(self, bonus_factor):

        if self.use_jit:

            # print(self.episode)
            # print(self.env.action_space.n)
            # print(self.horizon)
            # print(self.nstate_feat_hist)
            # print(self.hist_reward)
            # print(self.hist_rkhs_norm)
            # print(self.gram_mat_inv)
            # print(bonus_factor)
            # print(self.v_max)
            # print(self.gamma)

            gram_mat_inv = List()
            for mat in self.gram_mat_inv:
                gram_mat_inv.append(mat)

            reward_hist = List()
            for lst in self.hist_reward:
                reward_hist.append(lst)

            return run_kovi_jit(self.episode,
                                self.env.action_space.n,
                                self.horizon,
                                self.nstate_feat_hist,
                                reward_hist,
                                self.hist_rkhs_norm,
                                gram_mat_inv,
                                bonus_factor,
                                self.v_max,
                                self.gamma)

        else:
            q_ns = np.zeros((self.episode, self.env.action_space.n))
            alphas = np.zeros((self.horizon + 1, self.episode))

            for step in range(self.horizon - 1, -1, -1):
                # build targets
                prev_rewards = self._get_previous_rewards(step, by='episode')
                targets = prev_rewards + self.gamma * q_ns.max(axis=1)

                # update parameters solving kernel ridge regression
                alphas[step] = self.gram_mat_inv[step] @ targets
                self.alphas[step] = alphas[step]

                # update value function
                # prev_ns = self._get_previous_states(step + 1, by='episode')
                # prev_ns = [self.nstate_hist[step + ep * self.horizon] for ep in range(self.episode)]
                # q_ns = np.array([self._compute_q_vec(step + 1, ns, self.bonus_scale_factor) for ns in prev_ns])

                for tau in range(self.episode):
                    pns = self.hist_nstate[step][tau]
                    for a in range(self.env.action_space.n):
                        feat = self.nstate_feat_hist[step, tau, :self.episode, a]

                        # if a == 0:
                            # print(f'Episode {self.episode}, step {step}, tau {tau}, feat: {feat.shape}, gram_mat_inv[step]: {self.gram_mat_inv[step].shape}')
                        inv_counts = self.pd_kernel(pns, a) - feat @ self.gram_mat_inv[step] @ feat
                        bonus = bonus_factor * np.sqrt(inv_counts) + self.v_max * inv_counts * (bonus_factor > 0.0)
                        q_ns[tau, a] = feat.T @ self.alphas[step] + bonus
                        q_ns[tau, a] = min(q_ns[tau, a], self.v_max)

            return alphas


@numba_jit
def run_kovi_jit(episode,
                 n_actions,
                 horizon,
                 nstate_feat_hist,
                 reward_hist,
                 rkhs_norm_hist,
                 gram_mat_inv,
                 bonus_scale_factor,
                 v_max,
                 gamma):

    q_ns = np.zeros((episode, n_actions))
    alphas = np.zeros((horizon + 1, episode))
    targets = np.zeros(episode)

    for step in range(horizon - 1, -1, -1):
        prev_rewards = np.array(reward_hist[step])
        for ep in range(episode):
            targets[ep] = prev_rewards[ep] + gamma * max(q_ns[ep])

        alphas[step] = gram_mat_inv[step].dot(targets)

        for tau in range(episode):
            for a in range(n_actions):
                feat = nstate_feat_hist[step, tau, :episode, a]

                inv_counts = rkhs_norm_hist[step, tau, a] - feat.dot(gram_mat_inv[step].T).dot(feat)
                bonus = bonus_scale_factor * np.sqrt(inv_counts) + v_max * inv_counts * (bonus_scale_factor > 0.0)

                q_ns[tau, a] = feat.dot(alphas[step]) + bonus
                q_ns[tau, a] = min(q_ns[tau, a], v_max)

    return alphas
