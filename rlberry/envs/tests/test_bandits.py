import numpy as np
from rlberry.seeding import safe_reseed
from rlberry.seeding import Seeder
from rlberry.envs.bandits import (
    BernoulliBandit,
    NormalBandit,
    CorruptedNormalBandit,
)


TEST_SEED = 42


def test_bernoulli():
    env = BernoulliBandit(p=[0.05, 0.95])
    safe_reseed(env, Seeder(TEST_SEED))

    sample = [env.step(1)[1] for f in range(1000)]
    assert np.abs(np.mean(sample) - 0.95) < 0.1


def test_normal():
    env = NormalBandit(means=[0, 1])
    safe_reseed(env, Seeder(TEST_SEED))

    sample = [env.step(1)[1] for f in range(1000)]
    assert np.abs(np.mean(sample) - 1) < 0.1


def test_cor_normal():
    env = CorruptedNormalBandit(means=[0, 1], cor_prop=0.1)
    safe_reseed(env, Seeder(TEST_SEED))

    sample = [env.step(1)[1] for f in range(1000)]
    assert np.abs(np.median(sample) - 1) < 0.5
