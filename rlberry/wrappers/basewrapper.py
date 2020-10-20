from copy import deepcopy
from rlberry.envs.interface import ForwardModel, GenerativeModel, SimulationModel
from rlberry.rendering import BaseRenderInterface

class Wrapper(SimulationModel):
    """
    Wraps a given environment, similar to OpenAI gym's wrapper [1].

    It inherits from SimulationModel and BaseRenderInterface, but its behavior depends 
    on the wrapped enviroment (env):
    - If env does not implement step(), calling Wrapper.step() will raise an error.
    - If env does not implement sample(), calling Wrapper.sample() will raise an error.
    - If env does not implement render(), calling Wrapper.render() will raise an error

    [1] https://github.com/openai/gym/blob/master/gym/core.py
    """
    def __init__(self, env):
        self.env = deepcopy(env)
        self.observation_space = self.env.observation_space 
        self.action_space      = self.env.action_space
        self.reward_range      = self.env.reward_range

    def reset(self):
        if not isinstance(self.env, ForwardModel):
            raise NotImplementedError("Wrapped environment does not implemement reset().")
        return self.env.reset()

    def step(self, action):
        if not isinstance(self.env, ForwardModel):
            raise NotImplementedError("Wrapped environment does not implemement step().")
        return self.env.step(action)

    def sample(self, state, action):
        if not isinstance(self.env, GenerativeModel):
            raise NotImplementedError("Wrapped environment does not implemement sample().")
        return self.env.sample(state, action)
    
    def render(self, **kwargs):
        if not isinstance(self.env, BaseRenderInterface):
            raise NotImplementedError("Wrapped environment does not implemement render().")
        return self.env.render(**kwargs)
    
    def enable_rendering(self):
        if not isinstance(self.env, BaseRenderInterface):
            raise NotImplementedError("Wrapped environment does not implemement enable_rendering().") 
        self.env.enable_rendering()

    def disable_rendering(self):
        if not isinstance(self.env, BaseRenderInterface):
            raise NotImplementedError("Wrapped environment does not implemement disable_rendering().") 
        self.env.disable_rendering()

    @property
    def unwrapped(self):
        return self.env