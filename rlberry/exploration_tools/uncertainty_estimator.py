from abc import ABC, abstractmethod
import numpy as np


class UncertaintyEstimator(ABC):
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, **kwargs):
        pass

    @abstractmethod
    def update(self, state, action, next_state, reward, **kwargs):
        pass

    @abstractmethod
    def measure(self, state, action, **kwargs):
        pass

    def measure_batch(self, states, actions, **kwargs):
        return np.array([self.measure(s, a, **kwargs) for s, a in zip(states, actions)])
