from rlberry.seeding import safe_reseed
from rlberry.seeding import Seeder
import numpy as np
from rlberry.utils.check_gym_env import check_gym_env

seeder = Seeder(42)


def check_env(env):
    """
    Check that the environment is (almost) gym-compatible and that it is reproducible
    in the sense that it returns the same states when given the same seed.

    Parameters
    ----------
    env: gymnasium.env or rlberry env
        Environment that we want to check.
    """
    # Small reproducibility test
    action = env.action_space.sample()
    safe_reseed(env, Seeder(42))
    env.reset()
    a = env.step(action)[0]

    safe_reseed(env, Seeder(42))
    env.reset()
    b = env.step(action)[0]
    if hasattr(a, "__len__"):
        assert np.all(
            np.array(a) == np.array(b)
        ), "The environment does not seem to be reproducible"
    else:
        assert a == b, "The environment does not seem to be reproducible"

    # Modified check suite from gym
    check_gym_env(env)


def check_rlberry_env(env):
    """
    Companion to check_env, contains additional tests. It is not mandatory
    for an environment to satisfy this check but satisfying this check give access to
    additional features in rlberry.

    Parameters
    ----------
    env: gymnasium.env or rlberry env
        Environment that we want to check.
    """
    try:
        env.get_params()
    except Exception:
        raise RuntimeError("Fail to call get_params on the environment.")
