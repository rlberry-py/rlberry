from abc import ABC, abstractmethod
from copy import deepcopy
from rlberry.seeding import get_rng

class Agent(ABC):
    """
    Basic interface for agents.

    Attributes
    ----------
    id : string
        agent identifier
    env : rlberry.envs.interface.model.Model
        environment on which to train the agent
    
    Methods
    --------
    fit(**kwargs)
        train the agent, returns dictionary with training info
    policy(observation, **kwargs)
        returns the action to be taken given an observation
    reset()
        puts the agent in default setup (optional)
    """
    def __init__(self, env, **kwargs):
        self.id  = ""
        self.env = deepcopy(env)
        self.env.rng = get_rng()

    @abstractmethod
    def fit(self, **kwargs):
        pass 

    @abstractmethod
    def policy(self, observation, **kwargs):
        pass 

    def reset(self, **kwargs):
        pass 