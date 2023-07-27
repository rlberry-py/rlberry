from abc import ABC, abstractmethod
from rlberry.exploration_tools.typing import _get_type
import numpy as np


class UncertaintyEstimator(ABC):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : Keyword Arguments
            Extra argument (if necessary by inherited class)
        """
        pass

    @abstractmethod
    def update(self, state, action, next_state, reward, **kwargs):
        """
        Parameters
        ----------
        **kwargs : Keyword Arguments
            Extra argument (if necessary by inherited class)
        """
        pass

    @abstractmethod
    def measure(self, state, action, **kwargs):
        """
        Parameters
        ----------
        **kwargs : Keyword Arguments
            Extra argument (if necessary by inherited class)
        """
        pass

    def measure_batch(self, states, actions, **kwargs):
        batch = [self.measure(s, a, **kwargs) for s, a in zip(states, actions)]
        if _get_type(batch[0]) == "torch":
            import torch

            return torch.FloatTensor(batch)
        return np.array(batch)

    def measure_batch_all_actions(self, states):
        return np.array(
            [[self.measure(s, a) for a in range(self.action_space.n)] for s in states]
        )
