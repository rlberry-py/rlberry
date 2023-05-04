import gymnasium as gym
from rlberry.seeding import Seeder


class Discrete(gym.spaces.Discrete):
    """
    Class that represents discrete spaces.


    Inherited from gymnasium.spaces.Discrete for compatibility with gym.

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
        """
        Parameters
        ----------
        n : int
            number of elements in the space
        """
        assert n >= 0, "The number of elements in Discrete must be >= 0"
        gym.spaces.Discrete.__init__(self, n)
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
        return self.rng.integers(0, self.n)

    def __str__(self):
        objstr = "%d-element Discrete space" % self.n
        return objstr
