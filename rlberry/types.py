import gym
from typing import Any, Callable, Mapping, Tuple, Union

# either a gym.Env or a tuple containing (constructor, kwargs) to build the env
Env = Union[gym.Env, Tuple[Callable[..., gym.Env], Mapping[str, Any]]]
