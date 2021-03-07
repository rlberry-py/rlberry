import gym
from rlberry.seeding import Seeder


class MultiDiscrete(gym.spaces.MultiDiscrete, Seeder):
    """

    Inherited from gym.spaces.MultiDiscrete for compatibility with gym.

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
    def __init__(self, nvec):
        gym.spaces.MultiDiscrete.__init__(self, nvec)
        Seeder.__init__(self)

    def sample(self):
        sample = self.rng.random(self.nvec.shape)*self.nvec
        return sample.astype(self.dtype)
