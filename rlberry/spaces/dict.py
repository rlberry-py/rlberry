import gymnasium as gym
from rlberry.seeding import Seeder


class Dict(gym.spaces.Dict):
    """

    Inherited from gymnasium.spaces.Dict for compatibility with gym.

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

    def __init__(self, spaces=None, **spaces_kwargs):
        gym.spaces.Dict.__init__(self, spaces, **spaces_kwargs)
        self.seeder = Seeder()

    @property
    def rng(self):
        return self.seeder.rng

    def reseed(self, seed_seq=None):
        _ = [space.reseed(seed_seq) for space in self.spaces.values()]
