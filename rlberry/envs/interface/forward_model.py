from abc import abstractmethod
from rlberry.envs.interface.model import Model


class ForwardModel(Model):
    """
    Base class for foward models.

    A forward model allows us to sample trajectories from an environment.

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
    reset()
        puts the environment in a default state and returns this state
    step(action)
        returns the outcome of an action
    """
    def __init__(self):
        super(ForwardModel, self).__init__()

    @abstractmethod
    def reset(self):
        """
        Puts the environment in a default state and returns this state.
        """
        pass

    @abstractmethod
    def step(action):
        """
        Execute a step. Similar to gym function [1].
        [1] https://gym.openai.com/docs/#environments

        Parameters
        ----------
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
