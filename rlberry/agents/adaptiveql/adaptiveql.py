import gymnasium.spaces as spaces
import numpy as np
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.adaptiveql.tree import MDPTreePartition

import rlberry

logger = rlberry.logger


class AdaptiveQLAgent(AgentWithSimplePolicy):
    """
    Adaptive Q-Learning algorithm [1]_ implemented for enviroments
    with continuous (Box) states and **discrete actions**.

    .. todo:: Handle continuous actios too.

    Parameters
    ----------
    env : gym.Env
        Environment with continuous states and discrete actions.
    gamma : double, default: 1.0
        Discount factor in [0, 1].
    horizon : int, default: 50
        Horizon of the objective function.
    bonus_scale_factor : double, default: 1.0
        Constant by which to multiply the exploration bonus, controls
        the level of exploration.
    bonus_type : string, default: "simplified_bernstein"
        Type of exploration bonus. Currently, only "simplified_bernstein"
        is implemented.

    Attributes
    ----------
    gamma : double, default: 1.0
        Discount factor in [0, 1].
    horizon : int, default: 50
        Horizon of the objective function.
    bonus_scale_factor : double, default: 1.0
        Constant by which to multiply the exploration bonus, controls
        the level of exploration.
    bonus_type : string, default: "simplified_bernstein"
        Type of exploration bonus. Currently, only "simplified_bernstein"
        is implemented.
    v_max : ndarray
        Array of the maximum state value as a function of the Horizon.
    Qtree : MDPTreePartition
        Tree structure to represent the MDP model of transition.
    episode : int
        Number of episodes done during training of the adaptiveql agent.

    References
    ----------
    .. [1] Sinclair, Sean R., Siddhartha Banerjee, and Christina Lee Yu.
    "Adaptive Discretization for Episodic Reinforcement Learning in Metric Spaces."
     Proceedings of the ACM on Measurement and Analysis of Computing Systems 3.3 (2019): 1-44.

    Notes
    ------
    Uses the metric induced by the l-infinity norm.
    """

    name = "AdaptiveQLearning"

    def __init__(
        self,
        env,
        gamma=1.0,
        horizon=50,
        bonus_scale_factor=1.0,
        bonus_type="simplified_bernstein",
        **kwargs
    ):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        self.gamma = gamma
        self.horizon = horizon
        self.bonus_scale_factor = bonus_scale_factor
        self.bonus_type = bonus_type

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

        self.reset()

    def reset(self):
        self.Qtree = MDPTreePartition(
            self.env.observation_space, self.env.action_space, self.horizon
        )

        # info
        self.episode = 0

    def policy(self, observation):
        action, _ = self.Qtree.get_argmax_and_node(observation, 0)
        return action

    def _get_action_and_node(self, observation, hh):
        action, node = self.Qtree.get_argmax_and_node(observation, hh)
        return action, node

    def _update(self, node, state, action, next_state, reward, hh):
        # split node if necessary
        node_to_check = self.Qtree.update_counts(state, action, hh)
        if node_to_check.n_visits >= (self.Qtree.dmax / node_to_check.radius) ** 2.0:
            node_to_check.split()
        assert id(node_to_check) == id(node)

        tt = node.n_visits  # number of visits to the selected state-action node

        # value at next_state
        value_next_state = 0
        if hh < self.horizon - 1:
            value_next_state = min(
                self.v_max[hh + 1],
                self.Qtree.get_argmax_and_node(next_state, hh + 1)[1].qvalue,
            )

        # learning rate
        alpha = (self.horizon + 1.0) / (self.horizon + tt)

        bonus = self._compute_bonus(tt, hh)
        target = reward + bonus + self.gamma * value_next_state

        # update Q
        node.qvalue = (1 - alpha) * node.qvalue + alpha * target

    def _compute_bonus(self, n, hh):
        if self.bonus_type == "simplified_bernstein":
            bonus = self.bonus_scale_factor * np.sqrt(1.0 / n) + self.v_max[hh] / n
            bonus = min(bonus, self.v_max[hh])
            return bonus
        else:
            raise ValueError(
                "Error: bonus type {} not implemented".format(self.bonus_type)
            )

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        observation, info = self.env.reset()
        for hh in range(self.horizon):
            action, node = self._get_action_and_node(observation, hh)
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            done = terminated or truncated
            episode_rewards += reward

            self._update(node, observation, action, next_observation, reward, hh)

            observation = next_observation
            if done:
                break

        # update info
        self.episode += 1

        # writer
        if self.writer is not None:
            self.writer.add_scalar("episode_rewards", episode_rewards, self.episode)

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
