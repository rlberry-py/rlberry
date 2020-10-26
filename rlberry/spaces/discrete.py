import numpy as np

from rlberry.spaces import Space


class Discrete(Space):
    """
    Class that represents discrete spaces.

    Attributes
    ----------
    n : int
        number of elements in the space
    rng : numpy.random._generator.Generator
        random number generator provided by rlberry.seeding

    Methods
    -------
    sample()
        sample element from the space
    contains(x)
        check if x belongs to the space
    """

    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            number of elements in the space
        """
        assert n >= 0, "The number of elements in Discrete must be >= 0"
        super(Discrete, self).__init__()
        self.n = n

    def sample(self):
        return self.rng.integers(0, self.n)

    def contains(self, x):
        return (x >= 0) and (x < self.n) and np.issubdtype(type(x), np.integer)

    def __str__(self):
        objstr = "%d-element Discrete space" % self.n
        return objstr
