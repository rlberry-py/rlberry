from abc import ABC, abstractmethod
import numpy as np
from gym import spaces

from rlberry.seeding import seeding


class DiscreteDistribution(ABC):
    def __init__(self, **kwargs):
        self.np_random = None

    @abstractmethod
    def get_distribution(self):
        """
        Returns
        -------
        A distribution over actions {action:probability}
        """
        raise NotImplementedError()

    def sample(self):
        """
        Returns
        -------
        An action sampled from the distribution
        """
        distribution = self.get_distribution()
        return self.np_random.choice(
                    list(distribution.keys()), 1,
                    p=np.array(list(distribution.values())))[0]

    def seed(self):
        """
        Seed the policy randomness source
        """
        self.np_random = seeding.get_rng()

    def set_time(self, time):
        """
        Set the local time, allowing to schedule the distribution temperature.
        """
        pass

    def step_time(self):
        """
        Step the local time, allowing to schedule the distribution temperature.
        """
        pass


class EpsilonGreedy(DiscreteDistribution):
    """
    Uniform distribution with probability epsilon, and optimal action with
    probability 1-epsilon.
    """

    def __init__(self,
                 action_space,
                 temperature=1.0,
                 final_temperature=0.1,
                 tau=5000,
                 **kwargs):
        super().__init__(**kwargs)
        self.action_space = action_space
        self.temperature = temperature
        self.final_temperature = final_temperature
        self.tau = tau
        if isinstance(self.action_space, spaces.Tuple):
            self.action_space = self.action_space.spaces[0]
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("The action space should be discrete")
        self.final_temperature = min(self.temperature, self.final_temperature)
        self.optimal_action = None
        self.epsilon = 0
        self.time = 0
        self.writer = None
        self.seed()

    def get_distribution(self):
        distribution = {action: self.epsilon / self.action_space.n
                        for action in range(self.action_space.n)}
        distribution[self.optimal_action] += 1 - self.epsilon
        return distribution

    def update(self, values):
        """
        Update the action distribution parameters

        Parameters
        -----------
        values
            The state-action values
        step_time
            Whether to update epsilon schedule
        """
        self.optimal_action = np.argmax(values)
        self.epsilon = self.final_temperature \
            + (self.temperature - self.final_temperature) * \
            np.exp(- self.time / self.tau)
        if self.writer:
            self.writer.add_scalar('exploration/epsilon',
                                   self.epsilon,
                                   self.time)

    def step_time(self):
        self.time += 1

    def set_time(self, time):
        self.time = time

    def set_writer(self, writer):
        self.writer = writer


class Greedy(DiscreteDistribution):
    """
    Always use the optimal action
    """

    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space
        if isinstance(self.action_space, spaces.Tuple):
            self.action_space = self.action_space.spaces[0]
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("The action space should be discrete")
        self.values = None
        self.seed()

    def get_distribution(self):
        optimal_action = np.argmax(self.values)
        return {action: 1 if action == optimal_action
                else 0 for action in range(self.action_space.n)}

    def update(self, values):
        self.values = values


def exploration_factory(action_space, method="EpsilonGreedy", **kwargs):
    """
    Handles creation of exploration policies

    Parameters
    ----------
    exploration_config : dict
        Configuration dictionary of the policy, must contain a "method" key.
    action_space : gym.spaces.Space
        The environment action space

    Returns
    -------
    A new exploration policy.
    """
    if method == 'Greedy':
        return Greedy(action_space, **kwargs)
    elif method == 'EpsilonGreedy':
        return EpsilonGreedy(action_space, **kwargs)
    else:
        raise ValueError("Unknown exploration method")
