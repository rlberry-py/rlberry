import numpy as np


def representation_2d(state, env):
    # If state is 2D, just use it
    if isinstance(state, np.ndarray) and env.observation_space.shape == (2,):
        return state

    # Representation of highway-env environments
    try:
        from highway_env.envs.common.abstract import AbstractEnv
        from highway_env.envs.common.observation import KinematicObservation
        if isinstance(env.unwrapped, AbstractEnv) and \
                isinstance(env.unwrapped.observation_type, KinematicObservation):
            return np.array([state[0, KinematicObservation.FEATURES.index("x")],
                             -state[0, KinematicObservation.FEATURES.index("y")]])
    except ImportError:
        pass

    # No representation available
    return False
