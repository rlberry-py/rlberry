from abc import ABC, abstractmethod
from copy import deepcopy

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
    policy(state, **kwargs)
        returns the action to be taken in a given state
    reset()
        puts the agent in default setup (optional)
    """
    def __init__(self, env):
        self.id  = ""
        self.env = deepcopy(env)

    @abstractmethod
    def fit(self, **kwargs):
        pass 

    @abstractmethod
    def policy(self, state, **kwargs):
        pass 

    def reset(self):
        pass 