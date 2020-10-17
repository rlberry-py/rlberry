from abc import abstractmethod
from rlberry.envs.interface.model import Model


class GenerativeModel(Model):
    """
    Base class for generative models.

    A generative model contains a method to sample a transition from any
    state-action pair.

    Attributes
    ----------
    id : string
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
    --------
    sample(action, state)
        returns a transition sampled from taking an action in a given state
    """
    def __init__(self):
        super(GenerativeModel, self).__init__()

    @abstractmethod
    def sample(action, state):
        """
        Execute a step. Similar to gym function [1].
        [1] https://gym.openai.com/docs/#environments

        Parameters
        ----------
        state : object
            state from which to sample
        action : object
            action to take in the environment

        Returns:
        -------
        observation : object
        reward : float
        done  : bool
        info  : dict
        """
        pass
