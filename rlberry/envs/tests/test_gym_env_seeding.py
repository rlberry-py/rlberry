from rlberry.seeding.seeding import safe_reseed
import gym
import numpy as np
import pytest
from rlberry.seeding import Seeder
from rlberry.envs import gym_make

from copy import deepcopy

gym_envs = [
    'Acrobot-v1',
    'CartPole-v1',
    'MountainCar-v0',
]


def get_env_trajectory(env, horizon):
    states = []
    ss = env.reset()
    for ii in range(horizon):
        states.append(ss)
        ss, _, done, _ = env.step(env.action_space.sample())
        if done:
            ss = env.reset()
    return states


def compare_trajectories(traj1, traj2):
    for ss1, ss2 in zip(traj1, traj2):
        if not np.array_equal(ss1, ss2):
            return False
    return True


@pytest.mark.parametrize("env_name", gym_envs)
def test_env_seeding(env_name):
    seeder1 = Seeder(123)
    env1 = gym_make(env_name)
    env1.reseed(seeder1)

    seeder2 = Seeder(456)
    env2 = gym_make(env_name)
    env2.reseed(seeder2)

    seeder3 = Seeder(123)
    env3 = gym_make(env_name)
    env3.reseed(seeder3)

    if deepcopy(env1).is_online():
        traj1 = get_env_trajectory(env1, 500)
        traj2 = get_env_trajectory(env2, 500)
        traj3 = get_env_trajectory(env3, 500)

        assert not compare_trajectories(traj1, traj2)
        assert compare_trajectories(traj1, traj3)


@pytest.mark.parametrize("env_name", gym_envs)
def test_copy_reseeding(env_name):
    seeder = Seeder(123)
    env = gym_make(env_name)
    env.reseed(seeder)

    c_env = deepcopy(env)
    c_env.reseed()

    if deepcopy(env).is_online():
        traj1 = get_env_trajectory(env, 500)
        traj2 = get_env_trajectory(c_env, 500)
        assert not compare_trajectories(traj1, traj2)


@pytest.mark.parametrize("env_name", gym_envs)
def test_gym_safe_reseed(env_name):
    seeder = Seeder(123)
    seeder_aux = Seeder(123)

    env1 = gym.make(env_name)
    env2 = gym.make(env_name)
    env3 = gym.make(env_name)

    safe_reseed(env1, seeder)
    safe_reseed(env2, seeder)
    safe_reseed(env3, seeder_aux)

    traj1 = get_env_trajectory(env1, 500)
    traj2 = get_env_trajectory(env2, 500)
    traj3 = get_env_trajectory(env3, 500)
    assert not compare_trajectories(traj1, traj2)
    assert compare_trajectories(traj1, traj3)
