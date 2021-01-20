import numpy as np


def representation_2d(state, env):
    if isinstance(state, np.ndarray) and env.observation_space.shape == (2,):
        return state
    try:
        from highway_env.envs.common.abstract import AbstractEnv
        from highway_env.envs.common.observation import KinematicObservation
        if isinstance(env.unwrapped, AbstractEnv) and \
                isinstance(env.unwrapped.observation_type, KinematicObservation):
            return np.array([state[0, KinematicObservation.FEATURES.index("x")],
                            -state[0, KinematicObservation.FEATURES.index("y")]])
    except ImportError:
        pass
    return False
