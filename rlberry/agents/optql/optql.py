import logging
import numpy as np

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy
from rlberry.exploration_tools.discrete_counter import DiscreteCounter

logger = logging.getLogger(__name__)


class OptQLAgent(AgentWithSimplePolicy):
    """
    Optimistic Q-Learning [1]_ with custom exploration bonuses.

    Parameters
    ----------
    env : gym.Env
        Environment with discrete states and actions.
    gamma : double, default: 1.0
        Discount factor in [0, 1].
    horizon : int
        Horizon of the objective function.
    bonus_scale_factor : double, default: 1.0
        Constant by which to multiply the exploration bonus, controls
        the level of exploration.
    bonus_type : {"simplified_bernstein"}
        Type of exploration bonus. Currently, only "simplified_bernstein"
        is implemented.
    add_bonus_after_update : bool, default: False
        If True, add bonus to the Q function after performing the update,
        instead of adding it to the update target.

    References
    ----------
    .. [1] Jin et al., 2018
           Is Q-Learning Provably Efficient?
           https://arxiv.org/abs/1807.03765
    """
    name = "OptQL"

    def __init__(self,
                 env,
                 gamma=1.0,
                 horizon=100,
                 bonus_scale_factor=1.0,
                 bonus_type="simplified_bernstein",
                 add_bonus_after_update=False,
                 **kwargs):
        # init base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.gamma = gamma
        self.horizon = horizon
        self.bonus_scale_factor = bonus_scale_factor
        self.bonus_type = bonus_type
        self.add_bonus_after_update = add_bonus_after_update

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf or r_range == 0.0:
            logger.warning("{}: Reward range is  zero or infinity. ".format(self.name)
                           + "Setting it to 1.")
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

        # (s, a) visit counter
        self.N_sa = np.zeros((H, S, A))

        # Value functions
        self.V = np.ones((H + 1, S))
        self.V[H, :] = 0
        self.Q = np.ones((H, S, A))
        self.Q_bar = np.ones((H, S, A))
        for hh in range(self.horizon):
            self.V[hh, :] *= (self.horizon - hh)
            self.Q[hh, :, :] *= (self.horizon - hh)
            self.Q_bar[hh, :, :] *= (self.horizon - hh)

        if self.add_bonus_after_update:
            self.Q *= 0.0

        # ep counter
        self.episode = 0

        # useful object to compute total number of visited states & entropy of visited states
        self.counter = DiscreteCounter(self.env.observation_space,
                                       self.env.action_space)

    def policy(self, observation):
        """ Recommended policy. """
        state = observation
        return self.Q_bar[0, state, :].argmax()

    def _get_action(self, state, hh=0):
        """ Sampling policy. """
        return self.Q_bar[hh, state, :].argmax()

    def _compute_bonus(self, n, hh):
        if self.bonus_type == "simplified_bernstein":
            bonus = self.bonus_scale_factor * np.sqrt(1.0 / n) + self.v_max[hh] / n
            bonus = min(bonus, self.v_max[hh])
            return bonus
        else:
            raise ValueError(
                "Error: bonus type {} not implemented".format(self.bonus_type))

    def _update(self, state, action, next_state, reward, hh):
        self.N_sa[hh, state, action] += 1
        nn = self.N_sa[hh, state, action]

        # learning rate
        alpha = (self.horizon + 1.0) / (self.horizon + nn)
        bonus = self._compute_bonus(nn, hh)

        # bonus in the update
        if not self.add_bonus_after_update:
            target = reward + bonus + self.gamma * self.V[hh + 1, next_state]
            self.Q[hh, state, action] = (1 - alpha) * self.Q[hh, state, action] + alpha * target
            self.V[hh, state] = min(self.v_max[hh], self.Q[hh, state, :].max())
            self.Q_bar[hh, state, action] = self.Q[hh, state, action]
        # bonus outside the update
        else:
            target = reward + self.gamma * self.V[hh + 1, next_state]  # bonus not here
            self.Q[hh, state, action] = (1 - alpha) * self.Q[hh, state, action] + alpha * target
            self.Q_bar[hh, state, action] = self.Q[hh, state, action] + bonus  # bonus here
            self.V[hh, state] = min(self.v_max[hh], self.Q_bar[hh, state, :].max())

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        for hh in range(self.horizon):
            action = self._get_action(state, hh)
            next_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward  # used for logging only

            self.counter.update(state, action)

            self._update(state, action, next_state, reward, hh)

            state = next_state
            if done:
                break

        # update info
        self.episode += 1

        # writer
        if self.writer is not None:
            self.writer.add_scalar("episode_rewards", episode_rewards, self.episode)
            self.writer.add_scalar("n_visited_states", self.counter.get_n_visited_states(), self.episode)

        # return sum of rewards collected in the episode
        return episode_rewards

    def fit(self, budget: int, **kwargs):
        del kwargs
        n_episodes_to_run = budget
        count = 0
        while count < n_episodes_to_run:
            self._run_episode()
            count += 1
