import gym
import numpy as np
import logging
from rlberry.seeding import seeding

logger = logging.getLogger(__name__)


class Model(gym.Env):
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
    reseed()
        get new random number generator
    reset()
        puts the environment in a default state and returns this state
    step(action)
        returns the outcome of an action
    sample(state, action)
        returns a transition sampled from taking an action in a given state
    is_online()
        returns true if reset() and step() methods are implemented
    is_generative()
        returns true if sample() method is implemented
    """

    name = ""

    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self.reward_range: tuple = (-np.inf, np.inf)
        # random number generator
        self.rng = seeding.get_rng()

    def reseed(self):
        """
        Get new random number generator for the model.
        """
        self.rng = seeding.get_rng()
        self.observation_space.rng = self.rng
        self.action_space.rng = self.rng

    def sample(self, state, action):
        """
        Execute a step from a state-action pair.

        Parameters
        ----------
        state : object
            state from which to sample
        action : object
            action to take in the environment

        Returns
        -------
        observation : object
        reward : float
        done  : bool
        info  : dict
        """
        raise NotImplementedError("sample() method not implemented.")

    def is_online(self):
        logger.warning("Checking if Model is\
online calls reset() and step() methods.")
        try:
            self.reset()
            self.step(self.action_space.sample())
            return True
        except Exception as ex:
            if isinstance(ex, NotImplementedError):
                return False
            else:
                raise

    def is_generative(self):
        logger.warning("Checking if Model is \
generative calls sample() method.")
        try:
            self.sample(self.observation_space.sample(),
                        self.action_space.sample())
            return True
        except Exception as ex:
            if isinstance(ex, NotImplementedError):
                return False
            else:
                raise

    @property
    def unwrapped(self):
        return self
