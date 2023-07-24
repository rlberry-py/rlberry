import gymnasium as gym
import numpy as np
from rlberry.seeding import Seeder


class MultiDiscrete(gym.spaces.MultiDiscrete):
    """

    Inherited from gymnasium.spaces.MultiDiscrete for compatibility with gym.

    rlberry wraps gym.spaces to make sure the seeding
    mechanism is unified in the library (rlberry.seeding)

    Attributes
    ----------
    rng : numpy.random._generator.Generator
        random number generator provided by rlberry.seeding

    Methods
    -------
    reseed()
        get new random number generator
    """

    def __init__(self, nvec, dtype=np.int64):
        gym.spaces.MultiDiscrete.__init__(self, nvec, dtype=dtype)
        self.seeder = Seeder()

    @property
    def rng(self):
        return self.seeder.rng

    def reseed(self, seed_seq=None):
        """
        Get new random number generator.

        Parameters
        ----------
        seed_seq : np.random.SeedSequence, rlberry.seeding.Seeder or int, default : None
            Seed sequence from which to spawn the random number generator.
            If None, generate random seed.
            If int, use as entropy for SeedSequence.
            If seeder, use seeder.seed_seq
        """
        self.seeder.reseed(seed_seq)

    def sample(self):
        sample = self.rng.random(self.nvec.shape) * self.nvec
        return sample.astype(self.dtype)
