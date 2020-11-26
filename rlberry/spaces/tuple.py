import gym
from rlberry.spaces import SpaceSeeder


class Tuple(gym.spaces.Tuple, SpaceSeeder):
    """

    Inherited from gym.spaces.Tuple for compatibility with gym.

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
    def __init__(self, spaces):
        gym.spaces.Tuple.__init__(self, spaces)
        SpaceSeeder.__init__(self)

    def reseed(self):
        _ = [space.reseed() for space in self.spaces]
