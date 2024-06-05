import gymnasium as gym
from typing import Any, Callable, Mapping, Tuple, Union
from rlberry.seeding import Seeder

# either a gymnasium.Env or a tuple containing (constructor, kwargs) to build the env
Env = Union[gym.Env, Tuple[Callable[..., gym.Env], Mapping[str, Any]]]

#
Seed = Union[Seeder, int]
