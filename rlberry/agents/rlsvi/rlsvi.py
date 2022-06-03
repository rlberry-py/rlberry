import logging
import numpy as np

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.agents.dynprog.utils import (
    backward_induction_in_place,
    backward_induction_reward_sd,
    backward_induction_sd,
)

logger = logging.getLogger(__name__)


class RLSVIAgent(AgentWithSimplePolicy):
    """
    RLSVI algorithm from [1,2] with Gaussian noise.

    Notes
    -----
    The recommended policy after all the episodes is computed with the empirical
    MDP.
    The std of the noise is of the form:
    scale/sqrt(n)+ V_max/n
    as for simplified Bernstein bonuses.

    Parameters
    ----------
    env : gym.Env
        Environment with discrete states and actions.
    gamma : double, default: 1.0
        Discount factor in [0, 1]. If gamma is 1.0, the problem is set to
        be finite-horizon.
    horizon : int
        Horizon of the objective function. If None and gamma<1, set to
        1/(1-gamma).
    scale_std_noise : double, delfault: 1.0
        scale the std of the noise. At step h the std is
        scale_std_noise/sqrt(n)+(H-h+1)/n
    reward_free : bool, default: False
        If true, ignores rewards.
    stage_dependent : bool, default: False
        If true, assume that transitions and rewards can change with the stage h.

    References
    ----------
    .. [1] Osband et al., 2014
        Generalization and Exploration via Randomized Value Functions
        https://arxiv.org/abs/1402.0635

    .. [2] Russo, 2019
        Worst-Case Regret Bounds for Exploration via Randomized Value Functions
        https://arxiv.org/abs/1906.02870

    """

    name = "RLSVI"

    def __init__(
        self,
        env,
        gamma=1.0,
        horizon=100,
        scale_std_noise=1.0,
        reward_free=False,
        stage_dependent=False,
        **kwargs
    ):
        # init base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.gamma = gamma
        self.horizon = horizon
        self.scale_std_noise = scale_std_noise
        self.reward_free = reward_free
        self.stage_dependent = stage_dependent

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # other checks
        assert gamma >= 0 and gamma <= 1.0
        if self.horizon is None:
            assert gamma < 1.0, "If no horizon is given, gamma must be smaller than 1."
            self.horizon = int(np.ceil(1.0 / (1.0 - gamma)))

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf or r_range == 0.0:
            logger.warning(
                "{}: Reward range is  zero or infinity. ".format(self.name)
                + "Setting it to 1."
            )
            r_range = 1.0

        self.v_max = np.zeros(self.horizon)
        self.v_max[-1] = r_range
        for hh in reversed(range(self.horizon - 1)):
            self.v_max[hh] = r_range + self.gamma * self.v_max[hh + 1]

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

        # stds prior
        self.std1_sa = self.scale_std_noise * np.ones((H, S, A))
        self.std2_sa = np.ones((H, S, A))
        # visit counter
        self.N_sa = np.ones(shape_hsa)

        # MDP estimator
        self.R_hat = np.zeros(shape_hsa)
        self.P_hat = np.ones(shape_hsas) * 1.0 / S

        # Value functions
        self.V = np.zeros((H, S))
        self.Q = np.zeros((H, S, A))
        # for rec. policy
        self.V_policy = np.zeros((H, S))
        self.Q_policy = np.zeros((H, S, A))

        # Init V and variances
        for hh in range(self.horizon):
            self.std2_sa[hh, :, :] *= self.v_max[hh]

        # ep counter
        self.episode = 0

        # useful object to compute total number of visited states & entropy of visited states
        self.counter = DiscreteCounter(
            self.env.observation_space, self.env.action_space
        )

    def policy(self, observation):
        state = observation
        assert self.Q_policy is not None
        return self.Q_policy[0, state, :].argmax()

    def _get_action(self, state, hh=0):
        """Sampling policy."""
        assert self.Q is not None
        return self.Q[hh, state, :].argmax()

    def _update(self, state, action, next_state, reward, hh):
        if self.stage_dependent:
            self.N_sa[hh, state, action] += 1

            nn = self.N_sa[hh, state, action]
            prev_r = self.R_hat[hh, state, action]
            prev_p = self.P_hat[hh, state, action, :]

            self.R_hat[hh, state, action] = (
                1.0 - 1.0 / nn
            ) * prev_r + reward * 1.0 / nn

            self.P_hat[hh, state, action, :] = (1.0 - 1.0 / nn) * prev_p
            self.P_hat[hh, state, action, next_state] += 1.0 / nn

        else:
            self.N_sa[state, action] += 1

            nn = self.N_sa[state, action]
            prev_r = self.R_hat[state, action]
            prev_p = self.P_hat[state, action, :]

            self.R_hat[state, action] = (1.0 - 1.0 / nn) * prev_r + reward * 1.0 / nn

            self.P_hat[state, action, :] = (1.0 - 1.0 / nn) * prev_p
            self.P_hat[state, action, next_state] += 1.0 / nn

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        # stds scale/sqrt(n)+(H-h+1)/n
        std_sa = self.std1_sa / np.sqrt(self.N_sa) + self.std2_sa / self.N_sa
        noise_sa = self.rng.normal(self.R_hat, std_sa)
        # run backward noisy induction
        if self.stage_dependent:
            backward_induction_sd(
                self.Q,
                self.V,
                self.R_hat + noise_sa,
                self.P_hat,
                self.gamma,
                self.v_max[0],
            )
        else:
            backward_induction_reward_sd(
                self.Q,
                self.V,
                self.R_hat + noise_sa,
                self.P_hat,
                self.gamma,
                self.v_max[0],
            )

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

        # update info
        self.episode += 1

        # writer
        if self.writer is not None:
            self.writer.add_scalar("episode_rewards", episode_rewards, self.episode)
            self.writer.add_scalar(
                "n_visited_states", self.counter.get_n_visited_states(), self.episode
            )

        # return sum of rewards collected in the episode
        return episode_rewards

    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.

        Parameters
        ----------
        budget: int
            number of episodes. Each episode runs for self.horizon unless it
            enconters a terminal state in which case it stops early.
        """
        del kwargs
        n_episodes_to_run = budget
        count = 0
        while count < n_episodes_to_run:
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
                self.v_max[0],
            )
        else:
            backward_induction_in_place(
                self.Q_policy,
                self.V_policy,
                self.R_hat,
                self.P_hat,
                self.horizon,
                self.gamma,
                self.v_max[0],
            )
