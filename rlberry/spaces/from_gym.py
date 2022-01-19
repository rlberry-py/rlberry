import rlberry.spaces
import gym.spaces


def convert_space_from_gym(space):
    if isinstance(space, gym.spaces.Box) and (
        not isinstance(space, rlberry.spaces.Box)
    ):
        return rlberry.spaces.Box(
            space.low, space.high, shape=space.shape, dtype=space.dtype
        )
    if isinstance(space, gym.spaces.Discrete) and (
        not isinstance(space, rlberry.spaces.Discrete)
    ):
        return rlberry.spaces.Discrete(n=space.n)
    if isinstance(space, gym.spaces.MultiBinary) and (
        not isinstance(space, rlberry.spaces.MultiBinary)
    ):
        return rlberry.spaces.MultiBinary(n=space.n)
    if isinstance(space, gym.spaces.MultiDiscrete) and (
        not isinstance(space, rlberry.spaces.MultiDiscrete)
    ):
        return rlberry.spaces.MultiDiscrete(
            nvec=space.nvec,
            dtype=space.dtype,
        )
    if isinstance(space, gym.spaces.Tuple) and (
        not isinstance(space, rlberry.spaces.Tuple)
    ):
        return rlberry.spaces.Tuple(
            spaces=[convert_space_from_gym(sp) for sp in space.spaces]
        )
    if isinstance(space, gym.spaces.Dict) and (
        not isinstance(space, rlberry.spaces.Dict)
    ):
        converted_spaces = dict()
        for key in space.spaces:
            converted_spaces[key] = convert_space_from_gym(space.spaces[key])
        return rlberry.spaces.Dict(spaces=converted_spaces)

    return space
