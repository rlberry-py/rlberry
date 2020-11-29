import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward',
                         'next_state', 'terminal', 'info'))


class ReplayMemory(object):
    """
    Container that stores and samples transitions.
    """
    def __init__(self,
                 transition_type=Transition,
                 capacity=10000,
                 n_steps=1,
                 gamma=0.99,
                 **kwargs):
        self.capacity = int(capacity)
        self.transition_type = transition_type
        self.n_steps = n_steps
        self.gamma = gamma
        self.memory = []
        self.position = 0

    @classmethod
    def default_config(cls):
        return dict()

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.position = len(self.memory) - 1
        elif len(self.memory) > self.capacity:
            self.memory = self.memory[:self.capacity]
        # Faster than append and pop
        self.memory[self.position] = self.transition_type(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, collapsed=True):
        """
        Sample a batch of transitions.

        If n_steps is greater than one, the batch will be composed of
        lists of successive transitions.

        Parameters
        ----------
        batch_size: int
            Size of the batch
        collapsed : bool
            Whether successive transitions must be collapsed into one n-step
            transition.

        Returns
        -------
        The sampled batch
        """
        # FIXME: use general seeding
        if self.n_steps == 1:
            # Directly sample transitions
            return random.sample(self.memory, batch_size)
        else:
            # Sample initial transition indexes
            indexes = random.sample(range(len(self.memory)), batch_size)
            # Get the batch of n-consecutive-transitions starting
            # from sampled indexes
            all_transitions = [self.memory[i:i+self.n_steps] for i in indexes]
            # Collapse transitions
            return map(self.collapse_n_steps, all_transitions) if collapsed \
                else all_transitions

    def collapse_n_steps(self, transitions):
        """
        Collapse n transitions <s,a,r,s',t> of a trajectory into one
        transition <s0, a0, Sum(r_i), sp, tp>.

        We start from the initial state, perform the first action, and
        then the return estimate is formed by accumulating the discounted
        rewards along the trajectory until a terminal state or the end of the
        trajectory is reached.

        Parameters
        ----------
        transitions : list
             A list of n successive transitions

        Returns
        -------
            The corresponding n-step transition
        """
        state, action, cumulated_reward, next_state, done, info = \
            transitions[0]
        discount = 1
        for transition in transitions[1:]:
            if done:
                break
            else:
                _, _, reward, next_state, done, info = transition
                discount *= self.gamma
                cumulated_reward += discount*reward
        return state, action, cumulated_reward, next_state, done, info

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


class CEMMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.clear()

    def clear(self):
        self.size = 0
        self.states = []
        self.actions = []
        self.rewards = []

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.size += 1
        if self.size == self.max_size+1:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.size = self.max_size
