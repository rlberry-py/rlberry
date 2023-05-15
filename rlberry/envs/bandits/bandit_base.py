from collections import deque

from rlberry.envs.interface import Model
import rlberry.spaces as spaces
from rlberry.seeding import Seeder

import rlberry

logger = rlberry.logger


class Bandit(Model):
    """
    Base class for a stochastic multi-armed bandit.

    Parameters
    ----------
    laws: list of laws.
        laws of the arms. can either be a frozen scipy law or any class that
        has a method .rvs().

    **kwargs: keywords arguments
        additional arguments sent to :class:`~rlberry.envs.interface.Model`

    """

    name = ""

    def __init__(self, laws=[], **kwargs):
        Model.__init__(self, **kwargs)
        self.laws = laws
        self.n_arms = len(self.laws)
        self.action_space = spaces.Discrete(self.n_arms)

        # Pre-sample 10 samples
        self.rewards = [
            deque(self.laws[action].rvs(size=10, random_state=self.rng))
            for action in range(self.n_arms)
        ]
        self.n_rewards = [10] * self.n_arms

    def step(self, action):
        """
        Sample the reward associated to the action.
        """
        # test that the action exists
        assert action < self.n_arms

        # If the queue of reward for the action is empty, sample 2*size of old reward queue. Otherwise, pop from queue
        if self.rewards[action]:
            reward = self.rewards[action].pop()
        else:
            self.n_rewards[action] = 2 * self.n_rewards[action]
            self.rewards[action] = deque(
                self.laws[action].rvs(
                    size=self.n_rewards[action], random_state=self.rng
                )
            )
            reward = self.rewards[action].pop()

        done = True

        return 0, reward, done, {}

    def reseed(self, seed_seq=None):
        if seed_seq is None:
            self.seeder = self.seeder.spawn()
        else:
            self.seeder = Seeder(seed_seq)
        # spaces
        self.action_space.reseed(self.seeder.seed_seq)

        self.rewards = [
            deque(self.laws[action].rvs(size=10, random_state=self.rng))
            for action in range(self.n_arms)
        ]
        self.n_rewards = [10] * self.n_arms

    def reset(self):
        """
        Reset the environment to a default state.
        """
        return 0


class AdversarialBandit(Model):
    """
    Base class for a adversarial multi-armed bandit with oblivious
    opponent, i.e all rewards are fixed in advance at the start of the run.

    Parameters
    ----------
    rewards: list of rewards, shape (T, A).
        Possible rewards up to horizon T for each of the A arms.

    **kwargs: keywords arguments
        additional arguments sent to :class:`~rlberry.envs.interface.Model`

    """

    name = ""

    def __init__(self, rewards=[], **kwargs):
        Model.__init__(self, **kwargs)
        self.n_arms = rewards.shape[1]
        self.rewards = deque(rewards)
        self.action_space = spaces.Discrete(self.n_arms)

    def step(self, action):
        """
        Sample the reward associated to the action.
        """
        # test that the action exists
        assert action < self.n_arms

        rewards = self.rewards.popleft()
        reward = rewards[action]
        done = True

        return 0, reward, done, {}

    def reset(self):
        """
        Reset the environment to a default state.
        """
        return 0
