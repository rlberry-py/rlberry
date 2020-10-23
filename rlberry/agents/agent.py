from abc import ABC, abstractmethod
from copy import deepcopy

class Agent(ABC):
    """
    Basic interface for agents.

    Input environment is (deep)copied and reseeded.

    Attributes
    ----------
    id : string
        agent identifier
    env : rlberry.envs.interface.model.Model
        environment on which to train the agent
    fit_info : tuple
        tuple of strings containing the keys in the dictionary returned by fit()
    
    Methods
    --------
    fit(**kwargs)
        train the agent, returns dictionary with training info, 
        whose keys are strings
    policy(observation, **kwargs)
        returns the action to be taken given an observation
    reset()
        puts the agent in default setup (optional)
    save(), optional
        save agent 
    load(), optional
        load agent, returns an instance of the agent
    """
    def __init__(self, env, **kwargs):
        self.id  = ""
        self.env = deepcopy(env)
        self.env.reseed()
        self.fit_info = ()

    @abstractmethod
    def fit(self, **kwargs):
        pass 

    @abstractmethod
    def policy(self, observation, **kwargs):
        pass 

    def reset(self, **kwargs):
        pass 

    def save(self, **kwargs):
        pass

    def load(self, **kwargs):
        pass 
    