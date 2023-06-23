import gymnasium as gym
from rlberry.seeding import Seeder


class MultiBinary(gym.spaces.MultiBinary):
    """

    Inherited from gymnasium.spaces.MultiBinary for compatibility with gym.

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

    def __init__(self, n):
        gym.spaces.MultiBinary.__init__(self, n)
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
        return self.rng.integers(low=0, high=2, size=self.n, dtype=self.dtype)
