import logging
import gym.spaces as spaces
import numpy as np
from rlberry.agents import IncrementalAgent
from rlberry.utils.writers import PeriodicWriter
from rlberry.agents.adaptiveql.utils import QFunctionTreePartition


logger = logging.getLogger(__name__)


class AdaptiveQLAgent(IncrementalAgent):
    """
    Adaptive Q-Learning algorithm [1]_ implemented for enviroments
    with continuous (Box) states and **discrete actions**.

    .. todo:: Handle continuous actios too.

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

    .. [1] Sinclair, Sean R., Siddhartha Banerjee, and Christina Lee Yu.
    "Adaptive Discretization for Episodic Reinforcement Learning in Metric Spaces."
     Proceedings of the ACM on Measurement and Analysis of Computing Systems 3.3 (2019): 1-44.

    Notes
    ------

    Uses the metric induced by the l-infinity norm.
    """

    name = 'AdaptiveQLearning'

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
        self.Qtree = QFunctionTreePartition(self.env.observation_space,
                                            self.env.action_space,
                                            self.horizon)

        # info
        self._rewards = np.zeros(self.n_episodes)
        self.episode = 0

        # default writer
        self.writer = PeriodicWriter(self.name,
                                     log_every=5*logger.getEffectiveLevel())

    def policy(self, observation, hh=0, **kwargs):
        action, _ = self.Qtree.get_argmax_and_node(observation, hh)
        return action

    def _get_action_and_node(self, observation, hh):
        action, node = self.Qtree.get_argmax_and_node(observation, hh)
        return action, node

    def _update(self, node, state, action, next_state, reward, hh):
        self.Qtree.update(state, action, hh)
        tt = node.n_visits  # number of visits to the selected state-action node

        # value at next_state
        value_next_state = 0
        if hh < self.horizon-1:
            value_next_state = min(
                self.v_max[hh+1],
                self.Qtree.get_argmax_and_node(next_state, hh+1)[1].value
            )

        # learning rate
        alpha = (self.horizon+1.0)/(self.horizon + tt)

        bonus = self._compute_bonus(tt, hh)
        target = reward + bonus + self.gamma*value_next_state

        # update
        node.value = (1-alpha)*node.value + alpha*target

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
            action, node = self._get_action_and_node(state, hh)
            next_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward

            self._update(node, state, action, next_state, reward, hh)

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
    import matplotlib.pyplot as plt
    from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
    env = get_benchmark_env(level=2)

    agent = AdaptiveQLAgent(env, n_episodes=100, horizon=30)
    agent.fit()

    agent.Qtree.plot(0, 20)
    plt.show()
