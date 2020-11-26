import gym
from rlberry.spaces import SpaceSeeder


class Dict(gym.spaces.Dict, SpaceSeeder):
    """

    Inherited from gym.spaces.Dict for compatibility with gym.

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
        SpaceSeeder.__init__(self)

    def reseed(self):
        _ = [space.reseed() for space in self.spaces.values()]
