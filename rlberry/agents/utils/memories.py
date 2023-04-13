import numpy as np
from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "terminal", "info")
)


class ReplayMemory(object):
    """
    Container that stores and samples transitions.
    """

    def __init__(self, capacity=10000, **kwargs):
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0

    def push(self, item):
        """Saves a thing."""
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        # Faster than append and pop
        self.position = (self.position + 1) % self.capacity

    def _encode_sample(self, idxes):
        return [self.memory[idx] for idx in idxes]

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self))
        idxes = np.random.choice(len(self.memory), size=batch_size)
        return self._encode_sample(idxes), idxes

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) == self.capacity

    def is_empty(self):
        return len(self.memory) == 0


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
