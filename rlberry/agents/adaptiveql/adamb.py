import logging
import gym.spaces as spaces
import numpy as np
from rlberry.agents import IncrementalAgent
from rlberry.agents.adaptiveql.tree import MDPTreePartition
from rlberry.utils.writers import PeriodicWriter


logger = logging.getLogger(__name__)


class AdaMBAgent(IncrementalAgent):
    """
    Model-Based Reinforcement Learning with Adaptive Partitioning  [1]_

    .. warning::
        TO BE IMPLEMENTED, initially for enviroments with continuous (Box) states
        and **discrete actions**.

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

    .. [1]  Sinclair, S. R., Wang, T., Jain, G., Banerjee, S., & Yu, C. L. (2020).
           Adaptive Discretization for Model-Based Reinforcement Learning.
           arXiv preprint arXiv:2007.00717.

    Notes
    ------

    Uses the metric induced by the l-infinity norm.
    """

    name = 'AdaMBAgent'

    def __init__(self,
                 env,
                 n_episodes=1000,
                 gamma=1.0,
                 horizon=50,
                 bonus_scale_factor=1.0,
                 bonus_type="simplified_bernstein",
                 **kwargs):
        IncrementalAgent.__init__(self, env, **kwargs)

        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        self.n_episodes = n_episodes
        self.gamma = gamma
        self.horizon = horizon
        self.bonus_scale_factor = bonus_scale_factor
        self.bonus_type = bonus_type

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

        self.reset()

    def reset(self):
        # stores Q function and MDP model.
        self.model = MDPTreePartition(self.env.observation_space,
                                      self.env.action_space,
                                      self.horizon)

        # info
        self._rewards = np.zeros(self.n_episodes)
        self.episode = 0

        # default writer
        self.writer = PeriodicWriter(self.name,
                                     log_every=5*logger.getEffectiveLevel())

    def policy(self, observation, hh=0, **kwargs):
        return 0

    def _update(self, node, state, action, next_state, reward, hh):
        pass

    def _compute_bonus(self, n, hh):
        if self.bonus_type == "simplified_bernstein":
            bonus = self.bonus_scale_factor * np.sqrt(1.0 / n) + self.v_max[hh] / n
            bonus = min(bonus, self.v_max[hh])
            return bonus
        else:
            raise ValueError(
                "Error: bonus type {} not implemented".format(self.bonus_type))

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        for hh in range(self.horizon):
            action = 0    # TODO
            next_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward

            # self._update(node, state, action, next_state, reward, hh)

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


if __name__ == '__main__':
    from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env

    env = get_benchmark_env(level=2)
    agent = AdaMBAgent(env, n_episodes=50, horizon=30)
    agent.fit()
    agent.policy(env.observation_space.sample())
