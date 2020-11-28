import numpy as np
import pytest
import rlberry.seeding as seeding
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

    seeding.set_global_seed(123)
    env1 = gym_make(env_name)

    seeding.set_global_seed(456)
    env2 = gym_make(env_name)

    seeding.set_global_seed(123)
    env3 = gym_make(env_name)

    if deepcopy(env1).is_online():
        traj1 = get_env_trajectory(env1, 500)
        traj2 = get_env_trajectory(env2, 500)
        traj3 = get_env_trajectory(env3, 500)

        assert not compare_trajectories(traj1, traj2)
        assert compare_trajectories(traj1, traj3)


@pytest.mark.parametrize("env_name", gym_envs)
def test_copy_reseeding(env_name):

    seeding.set_global_seed(123)
    env = gym_make(env_name)

    c_env = deepcopy(env)
    c_env.reseed()

    if deepcopy(env).is_online():
        traj1 = get_env_trajectory(env, 500)
        traj2 = get_env_trajectory(c_env, 500)
        assert not compare_trajectories(traj1, traj2)
