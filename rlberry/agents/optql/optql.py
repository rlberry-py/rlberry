import logging
import numpy as np

import gym.spaces as spaces
from rlberry.agents import IncrementalAgent
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.utils.writers import PeriodicWriter


logger = logging.getLogger(__name__)


class OptQLAgent(IncrementalAgent):
    """
    Optimistic Q-Learning [1]_ with custom exploration bonuses.

    Parameters
    ----------
    env : gym.Env
        Environment with discrete states and actions.
    n_episodes : int
        Number of episodes
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

    References
    ----------
    [1] Jin et al., 2018
        Is Q-Learning Provably Efficient?
        https://arxiv.org/abs/1807.03765
    """
    name = "OptQL"
    fit_info = ("n_episodes", "episode_rewards")

    def __init__(self,
                 env,
                 n_episodes=1000,
                 gamma=1.0,
                 horizon=100,
                 bonus_scale_factor=1.0,
                 bonus_type="simplified_bernstein",
                 **kwargs):
        # init base class
        IncrementalAgent.__init__(self, env, **kwargs)

        self.n_episodes = n_episodes
        self.gamma = gamma
        self.horizon = horizon
        self.bonus_scale_factor = bonus_scale_factor
        self.bonus_type = bonus_type

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
        for hh in reversed(range(self.horizon-1)):
            self.v_max[hh] = r_range + self.gamma*self.v_max[hh+1]

        # initialize
        self.reset()

    def reset(self, **kwargs):
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n

        # (s, a) visit counter
        self.N_sa = np.zeros((H, S, A))

        # Value functions
        self.V = np.zeros((H+1, S))
        self.Q = np.ones((H, S, A))
        for hh in range(self.horizon):
            self.Q[hh, :, :] *= (self.horizon-hh)
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
        return self.Q[hh, state, :].argmax()

    def _get_action(self, state, hh=0):
        """ Sampling policy. """
        return self.Q[hh, state, :].argmax()

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
        alpha = (self.horizon+1.0)/(self.horizon + nn)
        bonus = self._compute_bonus(nn, hh)
        target = reward + bonus + self.gamma*self.V[hh+1, next_state]
        self.Q[hh, state, action] = (1-alpha)*self.Q[hh, state, action] + alpha * target
        self.V[hh, state] = min(self.v_max[hh], self.Q[hh, state, :].max())

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

        info = {"n_episodes": self.episode,
                "episode_rewards": self._rewards[:self.episode]}
        return info
