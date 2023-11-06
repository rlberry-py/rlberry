import gymnasium as gym
import numpy as np
from rlberry.seeding import Seeder


class Box(gym.spaces.Box):
    """
    Class that represents a space that is a cartesian product in R^n:

    [a_1, b_1] x [a_2, b_2] x ... x [a_n, b_n]


    Inherited from gymnasium.spaces.Box for compatibility with gym.

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

    def __init__(self, low, high, shape=None, dtype=np.float64):
        gym.spaces.Box.__init__(self, low, high, shape=shape, dtype=dtype)
        self.seeder = Seeder()

    @property
    def rng(self):
        return self.seeder.rng

    def reseed(self, seed_seq=None):
        """
        Get new random number generator.

        Parameters
        ----------
        seed_seq : np.random.SeedSequence, rlberry.seeding.Seeder or int, default : None
            Seed sequence from which to spawn the random number generator.
            If None, generate random seed.
            If int, use as entropy for SeedSequence.
            If seeder, use seeder.seed_seq
        """
        self.seeder.reseed(seed_seq)

    def sample(self):
        """
        Adapted from:
        https://raw.githubusercontent.com/openai/gym/master/gym/spaces/box.py


        Generates a single random sample inside of the Box.

        In creating a sample of the box, each coordinate is sampled according
        to the form of the interval:

        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1
        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = self.rng.normal(size=unbounded[unbounded].shape)

        sample[low_bounded] = (
            self.rng.exponential(size=low_bounded[low_bounded].shape)
            + self.low[low_bounded]
        )

        sample[upp_bounded] = (
            -self.rng.exponential(size=upp_bounded[upp_bounded].shape)
            + self.high[upp_bounded]
        )

        sample[bounded] = self.rng.uniform(
            low=self.low[bounded], high=high[bounded], size=bounded[bounded].shape
        )
        if self.dtype.kind == "i":
            sample = np.floor(sample)

        return sample.astype(self.dtype)
