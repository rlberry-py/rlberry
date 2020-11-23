from abc import ABC, abstractmethod


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
