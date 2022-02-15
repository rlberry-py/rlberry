from rlberry.seeding import safe_reseed
from rlberry.seeding import Seeder
import numpy as np
from rlberry.utils.check_gym_env import check_gym_env

seeder = Seeder(42)


def check_env(env):
    # Small reproducibility test
    action = env.action_space.sample()
    safe_reseed(env, Seeder(42))
    env.reset()
    a = env.step(action)[0]

    safe_reseed(env, Seeder(42))
    env.reset()
    b = env.step(action)[0]
    if hasattr(a, "__len__"):
        assert (
            np.mean(np.array(a) == np.array(b)) == 1
        ), "The environment does not seem to be reproducible"
    else:
        assert a == b, "The environment does not seem to be reproducible"

    # Modified check suite from gym
    check_gym_env(env)
