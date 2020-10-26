from abc import ABC, abstractmethod

from rlberry.seeding import seeding


class Space(ABC):
    """
    Base class to define observation and action spaces.

    Attributes
    ----------
    rng : numpy.random._generator.Generator
        random number generator provided by rlberry.seeding

    Methods
    -------
    sample()
        sample element from the space
    contains(x)
        check if x belongs to the space
    """

    def __init__(self):
        super().__init__()
        # random number generator
        self.rng = seeding.get_rng()

    @abstractmethod
    def sample(self):
        """
        Sample one element from the space.
        """
        pass

    @abstractmethod
    def contains(self, x):
        """
        Returns True if x belongs to the space, and False otherwise.
        """
        pass
