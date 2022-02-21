import numpy as np
from rlberry.seeding import safe_reseed
from rlberry.seeding import Seeder
from rlberry.envs.bandits import (
    AdversarialBandit,
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


def test_adversarial():
    r1 = np.concatenate((2 * np.ones((500, 1)), np.ones((500, 1))), axis=1)

    r2 = np.concatenate((np.ones((500, 1)), 2 * np.ones((500, 1))), axis=1)

    rewards = np.concatenate((r1, r2))

    env = AdversarialBandit(rewards=rewards)
    safe_reseed(env, Seeder(TEST_SEED))

    sample = [env.step(1)[1] for f in range(1000)]
    assert np.abs(np.mean(sample) - 1.5) < 1e-10
