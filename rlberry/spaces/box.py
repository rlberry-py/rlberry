import numpy as np
from rlberry.spaces import Space


class Box(Space):
    """
    Class that represents a space that is a cartesian product in R^n:

    [a_1, b_1] x [a_2, b_2] x ... x [a_n, b_n]

    Attributes
    ----------
    low : numpy.ndarray
         [a_1, ..., a_n]
    high : numpy.ndarray
         [b_1, ..., b_n]
    dim : int
        dimension of the space (n in R^n)
    rng : numpy.random._generator.Generator
        random number generator provided by rlberry.seeding

    Methods
    -------
    sample()
        sample element from the space
    contains(x)
        check if x belongs to the space
    """

    def __init__(self, low, high, dim=None, default_dtype=np.float64):
        """
        Parameters
        -----------
        low : numpy.ndarray or double
            - if low is a double, the parameter dim must be given, and
            low will be set to np.full((dim,), low, dtype=default_dtype)
            - if low is a np.ndarray, default_dtype is ignored
        high : numpy.ndarray or double
            - if high is a double, the parameter dim must be given, and
            high will be set to np.full((dim,), high, dtype=default_dtype)
            - if high is a np.ndarray, default_dtype is ignored
        dim :
            dimension of the space, to be given if low and high are double
        default_dtype :
            default type for values in numpy arrays, used if low and high are
            double
        """
        super(Box, self).__init__()
        assert type(low) is type(high), "low and high must have the same type"

        if type(low) is not np.ndarray:
            assert dim is not None and dim > 0, \
            "dim must be given if low/high are not arrays"
            self.low  = np.full((dim,), low,  dtype=default_dtype)
            self.high = np.full((dim,), high, dtype=default_dtype)
            self.dim  = dim
        else:
            assert low.ndim == high.ndim == 1, "high and low must be 1d"
            assert low.shape == high.shape, "low and high must have the same shape"
            self.low  = low.copy()
            self.high = high.copy()
            self.dim  = low.shape[0]

        # check which dimensions are bounded
        self._boundedness = np.zeros(self.dim, dtype=np.uint8)
        for dd in range(self.dim):
            assert self.low[dd] < self.high[dd], "low < high is required!"
            bounded_below = 1 * (self.low[dd]  != -np.inf)
            bounded_above = 1 * (self.high[dd] !=  np.inf)
            # 0 if not below and not above
            # 1 if not below and     above
            # 2 if     below and not above
            # 3 if     below and     above
            self._boundedness[dd] = 2*bounded_below +  bounded_above


    def sample(self):
        """
        For each dimension i:
        - if bounded above and below:   uniform distribution on [a_i, b_i]
        - if unbounded above:           exponential distribution on [a_i, infty]
        - if unbounded below:           exponential distribution on [-infty, b_i]
        - if unbounded above and below: normal distribution
        """
        xsample = np.zeros(self.dim)
        for dd in range(self.dim):
            # bounded above and below
            if   self._boundedness[dd] == 3:
                xsample[dd] = self.rng.uniform(self.low[dd], self.high[dd])
            # unbounded above
            elif self._boundedness[dd] == 2:
                xsample[dd] = self.low[dd] + self.rng.exponential()
            # unbounded below
            elif self._boundedness[dd] == 1:
                xsample[dd] = self.high[dd] - self.rng.exponential()
            # unbounded above and below
            elif self._boundedness[dd] == 0:
                xsample[dd] = self.rng.normal()
        return xsample

    def contains(self, x):
        contain = True
        contain = contain and (type(x) is np.ndarray)
        contain = contain and (x.ndim == 1)
        contain = contain and (x.shape[0] == self.dim)
        if contain:
            for dd in range(self.dim):
                contain = contain and \
                        (x[dd] <= self.high[dd]) and (x[dd] >= self.low[dd])
        return contain

    def __str__(self):
        objstr = "%d-dimensional Box space:\n"%self.dim
        for dd in range(self.dim):
            objstr += "   [%0.2f, %0.2f]\n"%(self.low[dd], self.high[dd])
        return objstr
