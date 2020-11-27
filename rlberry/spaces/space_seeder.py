from rlberry.seeding import seeding


class SpaceSeeder:
    """
    Base class to define observation and action spaces.

    Inherited from gym.spaces.Space for compatibility.

    rlberry wraps gym.spaces to make sure the seeding
    mechanism is unified in the library.

    Attributes
    ----------
    rng : numpy.random._generator.Generator
        random number generator provided by rlberry.seeding

    Methods
    -------
    reseed()
        get new random number generator
    """

    def __init__(self):
        super().__init__()
        # random number generator
        self.rng = seeding.get_rng()

    def reseed(self):
        self.rng = seeding.get_rng()
