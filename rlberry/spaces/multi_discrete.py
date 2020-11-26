import gym
from rlberry.spaces import SpaceSeeder


class MultiDiscrete(gym.spaces.MultiDiscrete, SpaceSeeder):
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
        SpaceSeeder.__init__(self)

    def sample(self):
        sample = self.rng.random(self.nvec.shape)*self.nvec
        return sample.astype(self.dtype)
