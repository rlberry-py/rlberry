from abc import ABC

import numpy as np

from rlberry.seeding import seeding
from rlberry.spaces import Space


class Model(ABC):
    """
    Base class for an environment model.

    Attributes
    ----------
    name : string
        environment identifier
    observation_space : rlberry.spaces.Space
        observation space
    action_space : rlberry.spaces.Space
        action space
    reward_range : tuple
        tuple (r_min, r_max) containing the minimum and the maximum reward
    rng : numpy.random._generator.Generator
        random number generator provided by rlberry.seeding 

    Methods
    -------
    reseed 
        get new random number generator
    """

    name = ""

    def __init__(self):
        super(Model, self).__init__()
        self.observation_space: Space = None
        self.action_space: Space = None
        self.reward_range: tuple = (-np.inf, np.inf)
        # random number generator
        self.rng = seeding.get_rng()

    def reseed(self):
        """
        Get new random number generator for the model.
        """
        self.rng = seeding.get_rng()
