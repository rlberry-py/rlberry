import gym
import numpy as np
import logging
import inspect
from collections import defaultdict
from rlberry.seeding import Seeder

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
    seeder : rlberry.seeding.Seeder
        Seeder, containing random number generator.

    Methods
    -------
    reseed(seed_seq)
        get new Seeder
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
        self.seeder = Seeder()

    def reseed(self, seed_seq=None):
        """
        Get new random number generator for the model.

        Parameters
        ----------
        seed_seq : np.random.SeedSequence, rlberry.seeding.Seeder or int, default : None
            Seed sequence from which to spawn the random number generator.
            If None, generate random seed.
            If int, use as entropy for SeedSequence.
            If seeder, use seeder.seed_seq
        """
        # self.seeder
        if seed_seq is None:
            self.seeder = self.seeder.spawn()
        else:
            self.seeder = Seeder(seed_seq)
        # spaces
        self.observation_space.reseed(self.seeder.seed_seq)
        self.action_space.reseed(self.seeder.seed_seq)

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
        logger.warning(
            "Checking if Model is\
online calls reset() and step() methods."
        )
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
        logger.warning(
            "Checking if Model is \
generative calls sample() method."
        )
        try:
            self.sample(self.observation_space.sample(), self.action_space.sample())
            return True
        except Exception as ex:
            if isinstance(ex, NotImplementedError):
                return False
            else:
                raise

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the Model"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]

        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this model.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this model and
            contained subobjects.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this model.
        The method works on simple estimators as well as on nested objects.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it'spossible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            parameters.
        Returns
        -------
        self : Model instance
            Model instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for environment %s. "
                    "Check the list of available parameters "
                    "with `env.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    @property
    def unwrapped(self):
        return self

    @property
    def rng(self):
        """Random number generator."""
        return self.seeder.rng
