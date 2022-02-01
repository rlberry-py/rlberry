from rlberry.seeding import safe_reseed
from rlberry.seeding import Seeder
import numpy as np

seeder = Seeder(42)


def check_env(env):
    # Test if env adheres to Gym API
    action = env.action_space.sample()
    safe_reseed(env, Seeder(42))
    env.reset()
    a = env.step(action)[0]

    safe_reseed(env, Seeder(42))
    env.reset()
    b = env.step(action)[0]
    if hasattr(a, "__len__"):
        print(a, b)
        assert np.mean(np.array(a) == np.array(b)) == 1
    else:
        assert a == b
