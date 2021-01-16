import logging
import numpy as np

import gym.spaces as spaces
from rlberry.agents import IncrementalAgent
from rlberry.agents.ucbvi.utils import update_value_and_get_action, update_value_and_get_action_sd
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.agents.dynprog.utils import backward_induction_sd
from rlberry.agents.dynprog.utils import backward_induction_in_place
from rlberry.utils.writers import PeriodicWriter


logger = logging.getLogger(__name__)


class UCBVIAgent(IncrementalAgent):
    """
    UCBVI [1]_ with custom exploration bonus.

    Notes
    -----
    The recommended policy after all the episodes is computed without
    exploration bonuses.

    Parameters
    ----------
    env : gym.Env
        Environment with discrete states and actions.
    n_episodes : int
        Number of episodes.
    gamma : double, default: 1.0
        Discount factor in [0, 1]. If gamma is 1.0, the problem is set to
        be finite-horizon.
    horizon : int
        Horizon of the objective function. If None and gamma<1, set to
        1/(1-gamma).
    bonus_scale_factor : double, default: 1.0
        Constant by which to multiply the exploration bonus, controls
        the level of exploration.
    bonus_type : {"simplified_bernstein"}
        Type of exploration bonus. Currently, only "simplified_bernstein"
        is implemented. If `reward_free` is true, this parameter is ignored
        and the algorithm uses 1/n bonuses.
    reward_free : bool, default: False
        If true, ignores rewards and uses only 1/n bonuses.
    stage_dependent : bool, default: False
        If true, assume that transitions and rewards can change with the stage h.
    real_time_dp : bool, default: False
        If true, uses real-time dynamic programming [2]_ instead of full backward induction
        for the sampling policy.

    References
    ----------
    .. [1] Azar et al., 2017
        Minimax Regret Bounds for Reinforcement Learning
        https://arxiv.org/abs/1703.05449

    .. [2] Efroni, Yonathan, et al.
          Tight regret bounds for model-based reinforcement learning with greedy policies.
          Advances in Neural Information Processing Systems. 2019.
          https://papers.nips.cc/paper/2019/file/25caef3a545a1fff2ff4055484f0e758-Paper.pdf
    """
    name = "UCBVI"

    def __init__(self,
                 env,
                 n_episodes=1000,
                 gamma=1.0,
                 horizon=100,
                 bonus_scale_factor=1.0,
                 bonus_type="simplified_bernstein",
                 reward_free=False,
                 stage_dependent=False,
                 real_time_dp=False,
                 **kwargs):
        # init base class
        IncrementalAgent.__init__(self, env, **kwargs)

        self.n_episodes = n_episodes
        self.gamma = gamma
        self.horizon = horizon
        self.bonus_scale_factor = bonus_scale_factor
        self.bonus_type = bonus_type
        self.reward_free = reward_free
        self.stage_dependent = stage_dependent
        self.real_time_dp = real_time_dp

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # other checks
        assert gamma >= 0 and gamma <= 1.0
        if self.horizon is None:
            assert gamma < 1.0, \
                "If no horizon is given, gamma must be smaller than 1."
            self.horizon = int(np.ceil(1.0 / (1.0 - gamma)))

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf or r_range == 0.0:
            logger.warning("{}: Reward range is  zero or infinity. ".format(self.name)
                           + "Setting it to 1.")
            r_range = 1.0

        self.v_max = np.zeros(self.horizon)
        self.v_max[-1] = r_range
        for hh in reversed(range(self.horizon-1)):
            self.v_max[hh] = r_range + self.gamma*self.v_max[hh+1]

        # initialize
        self.reset()

    def reset(self, **kwargs):
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n

        if self.stage_dependent:
            shape_hsa = (H, S, A)
            shape_hsas = (H, S, A, S)
        else:
            shape_hsa = (S, A)
            shape_hsas = (S, A, S)

        # (s, a) visit counter
        self.N_sa = np.zeros(shape_hsa)
        # (s, a) bonus
        self.B_sa = np.ones(shape_hsa)

        # MDP estimator
        self.R_hat = np.zeros(shape_hsa)
        self.P_hat = np.ones(shape_hsas) * 1.0/S

        # Value functions
        self.V = np.ones((H, S))
        self.Q = np.zeros((H, S, A))
        # for rec. policy
        self.V_policy = np.zeros((H, S))
        self.Q_policy = np.zeros((H, S, A))

        # Init V and bonus
        if not self.stage_dependent:
            self.B_sa *= self.v_max[0]
            self.V *= self.v_max[0]
        else:
            for hh in range(self.horizon):
                self.B_sa[hh, :, :] = self.v_max[hh]
                self.V[hh, :] = self.v_max[hh]

        # ep counter
        self.episode = 0

        # useful object to compute total number of visited states & entropy of visited states
        self.counter = DiscreteCounter(self.env.observation_space,
                                       self.env.action_space)

        # info
        self._rewards = np.zeros(self.n_episodes)

        # default writer
        self.writer = PeriodicWriter(self.name,
                                     log_every=5*logger.getEffectiveLevel())

    def policy(self, state, hh=0, **kwargs):
        """ Recommended policy. """
        assert self.Q_policy is not None
        return self.Q_policy[hh, state, :].argmax()

    def _get_action(self, state, hh=0):
        """ Sampling policy. """
        if not self.real_time_dp:
            assert self.Q is not None
            return self.Q[hh, state, :].argmax()
        else:
            if self.stage_dependent:
                update_fn = update_value_and_get_action_sd
            else:
                update_fn = update_value_and_get_action
            return update_fn(
                state,
                hh,
                self.V,
                self.R_hat,
                self.P_hat,
                self.B_sa,
                self.gamma,
                self.v_max,
                )

    def _compute_bonus(self, n, hh):
        # reward-free
        if self.reward_free:
            bonus = 1.0 / n
            return bonus

        # not reward-free
        if self.bonus_type == "simplified_bernstein":
            bonus = self.bonus_scale_factor * np.sqrt(1.0 / n) + self.v_max[hh] / n
            bonus = min(bonus, self.v_max[hh])
            return bonus
        else:
            raise ValueError(
                "Error: bonus type {} not implemented".format(self.bonus_type))

    def _update(self, state, action, next_state, reward, hh):
        if self.stage_dependent:
            self.N_sa[hh, state, action] += 1

            nn = self.N_sa[hh, state, action]
            prev_r = self.R_hat[hh, state, action]
            prev_p = self.P_hat[hh, state, action, :]

            self.R_hat[hh, state, action] = (1.0-1.0/nn)*prev_r + reward*1.0/nn

            self.P_hat[hh, state, action, :] = (1.0-1.0/nn)*prev_p
            self.P_hat[hh, state, action, next_state] += 1.0/nn

            self.B_sa[hh, state, action] = self._compute_bonus(nn, hh)

        else:
            self.N_sa[state, action] += 1

            nn = self.N_sa[state, action]
            prev_r = self.R_hat[state, action]
            prev_p = self.P_hat[state, action, :]

            self.R_hat[state, action] = (1.0-1.0/nn)*prev_r + reward*1.0/nn

            self.P_hat[state, action, :] = (1.0-1.0/nn)*prev_p
            self.P_hat[state, action, next_state] += 1.0/nn

            self.B_sa[state, action] = self._compute_bonus(nn, 0)

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        for hh in range(self.horizon):
            action = self._get_action(state, hh)
            next_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward  # used for logging only

            self.counter.update(state, action)

            if self.reward_free:
                reward = 0.0  # set to zero before update if reward_free

            self._update(state, action, next_state, reward, hh)

            state = next_state
            if done:
                break

        # run backward induction
        if not self.real_time_dp:
            if self.stage_dependent:
                backward_induction_sd(
                    self.Q,
                    self.V,
                    self.R_hat+self.B_sa,
                    self.P_hat,
                    self.gamma,
                    self.v_max[0])
            else:
                backward_induction_in_place(
                    self.Q,
                    self.V,
                    self.R_hat + self.B_sa,
                    self.P_hat,
                    self.horizon,
                    self.gamma,
                    self.v_max[0])

        # update info
        ep = self.episode
        self._rewards[ep] = episode_rewards
        self.episode += 1

        # writer
        if self.writer is not None:
            self.writer.add_scalar("ep reward", episode_rewards, self.episode)
            self.writer.add_scalar("n_visited_states", self.counter.get_n_visited_states(), self.episode)

        # return sum of rewards collected in the episode
        return episode_rewards

    def partial_fit(self, fraction, **kwargs):
        assert 0.0 < fraction <= 1.0
        n_episodes_to_run = int(np.ceil(fraction*self.n_episodes))
        count = 0
        while count < n_episodes_to_run and self.episode < self.n_episodes:
            self._run_episode()
            count += 1

        # compute Q function for the recommended policy
        if self.stage_dependent:
            backward_induction_sd(
                self.Q_policy,
                self.V_policy,
                self.R_hat,
                self.P_hat,
                self.gamma,
                self.v_max[0])
        else:
            backward_induction_in_place(
                self.Q_policy,
                self.V_policy,
                self.R_hat,
                self.P_hat,
                self.horizon,
                self.gamma,
                self.v_max[0])

        info = {"n_episodes": self.episode,
                "episode_rewards": self._rewards[:self.episode]}
        return info
