from typing import Optional, Literal
import numpy as np
from gymnasium import spaces
from scipy.special import softmax

from rlberry import types
from rlberry.agents import AgentWithSimplePolicy


class SARSAAgent(AgentWithSimplePolicy):
    """SARSA Agent.

    Parameters
    ----------
    env: :class:`~rlberry.types.Env`
        Environment with discrete states and actions.
    gamma: float, default = 0.99
        Discount factor.
    alpha: float, default = 0.1
        Learning rate.
    exploration_type: {"epsilon", "boltzmann"}, default: None
        If "epsilon": Epsilon-Greedy exploration.
        If "boltzmann": Boltzmann exploration.
        If None: No exploration.
    exploration_rate: float, default: None
        epsilon parameter for Epsilon-Greedy exploration or tau parameter for Boltzmann exploration.

    Attributes
    ----------
    Q : ndarray
        2D array that stores the estimation ofexpected rewards for state-action pairs.
    Examples
    --------
    >>> from rlberry.envs import GridWorld
    >>>
    >>> env = GridWorld(walls=(), nrows=5, ncols=5)
    >>> agent = SARSAAgent()
    >>> agent.fit(budget=1000)
    >>> agent.policy(env.observation_space.sample())
    >>> agent.reset()
    """

    def __init__(
        self,
        env: types.Env,
        gamma: float = 0.99,
        alpha: float = 0.1,
        exploration_type: Optional[Literal["epsilon", "boltzmann"]] = None,
        exploration_rate: Optional[float] = None,
        **kwargs
    ):
        # init base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.gamma = gamma
        self.alpha = alpha
        self.exploration_type = exploration_type
        self.exploration_rate = exploration_rate
        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # check exploration type
        if self.exploration_type is not None:
            assert (
                exploration_type == "epsilon" or "boltzmann"
            ) and exploration_rate is not None

        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def reset(self, **kwargs):
        self.Q.fill(0)

    def policy(self, observation):
        return self.Q[observation].argmax()

    def get_action(self, observation):
        if (
            self.exploration_type == "epsilon"
            and np.random.random() <= self.exploration_rate
        ):
            return np.random.choice(self.env.action_space.n)
        elif self.exploration_type == "boltzmann":
            return np.random.choice(
                self.env.action_space.n,
                p=softmax(self.exploration_rate * self.Q[observation]),
            )
        else:
            return self.Q[observation].argmax()

    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.
        Parameters
        ----------
        budget: int
            number of Q updates.
        """
        del kwargs
        observation, info = self.env.reset()
        episode_rewards = 0
        for i in range(budget):
            action = self.get_action(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            done = terminated or truncated
            episode_rewards += reward
            if self.writer is not None:
                self.writer.add_scalar("episode_rewards", episode_rewards, i)
            if done:
                self.Q[observation, action] = reward
            else:
                next_action = self.get_action(next_observation)
                self.Q[observation, action] = self.Q[
                    observation, action
                ] + self.alpha * (
                    reward
                    + self.gamma * self.Q[next_observation, next_action]
                    - self.Q[observation, action]
                )
            observation = next_observation
            if done:
                observation, info = self.env.reset()
                episode_rewards = 0
