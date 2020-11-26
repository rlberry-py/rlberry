import gym
from rlberry.spaces import SpaceSeeder


class Discrete(gym.spaces.Discrete, SpaceSeeder):
    """
    Class that represents discrete spaces.


    Inherited from gym.spaces.Discrete for compatibility with gym.

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
        SpaceSeeder.__init__(self)
        gym.spaces.Discrete.__init__(self, n)

    def sample(self):
        return self.rng.integers(0, self.n)

    def __str__(self):
        objstr = "%d-element Discrete space" % self.n
        return objstr
