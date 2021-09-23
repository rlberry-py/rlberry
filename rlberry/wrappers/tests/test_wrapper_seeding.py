import numpy as np
import pytest
from rlberry.seeding import Seeder

from copy import deepcopy
from rlberry.envs.classic_control import MountainCar, Acrobot
from rlberry.envs.finite import Chain
from rlberry.envs.finite import GridWorld
from rlberry.envs.benchmarks.ball_exploration import PBall2D, SimplePBallND
from rlberry.envs import Wrapper
from rlberry.wrappers import RescaleRewardWrapper

_GYM_INSTALLED = True
try:
    import gym
except Exception:
    _GYM_INSTALLED = False

classes = [
    MountainCar,
    GridWorld,
    Chain,
    PBall2D,
    SimplePBallND,
    Acrobot
]


def get_env_trajectory(env, horizon):
    states = []
    ss = env.reset()
    for _ in range(horizon):
        states.append(ss)
        ss, _, _, _ = env.step(env.action_space.sample())
    return states


def compare_trajectories(traj1, traj2):
    for ss1, ss2 in zip(traj1, traj2):
        if not np.array_equal(ss1, ss2):
            return False
    return True


@pytest.mark.parametrize("ModelClass", classes)
def test_wrapper_seeding(ModelClass):
    env1 = Wrapper(ModelClass())
    seeder = Seeder(123)
    env1.reseed(seeder)

    env2 = Wrapper(ModelClass())
    seeder = Seeder(456)
    env2.reseed(seeder)

    env3 = Wrapper(ModelClass())
    seeder = Seeder(123)
    env3.reseed(seeder)

    if deepcopy(env1).is_online():
        traj1 = get_env_trajectory(env1, 500)
        traj2 = get_env_trajectory(env2, 500)
        traj3 = get_env_trajectory(env3, 500)

        assert not compare_trajectories(traj1, traj2)
        assert compare_trajectories(traj1, traj3)


@pytest.mark.parametrize("ModelClass", classes)
def test_rescale_wrapper_seeding(ModelClass):
    env1 = RescaleRewardWrapper(ModelClass(), (0, 1))
    seeder = Seeder(123)
    env1.reseed(seeder)

    env2 = RescaleRewardWrapper(ModelClass(), (0, 1))
    seeder = Seeder(456)
    env2.reseed(seeder)

    env3 = RescaleRewardWrapper(ModelClass(), (0, 1))
    seeder = Seeder(123)
    env3.reseed(seeder)

    if deepcopy(env1).is_online():
        traj1 = get_env_trajectory(env1, 500)
        traj2 = get_env_trajectory(env2, 500)
        traj3 = get_env_trajectory(env3, 500)

        assert not compare_trajectories(traj1, traj2)
        assert compare_trajectories(traj1, traj3)


@pytest.mark.parametrize("ModelClass", classes)
def test_wrapper_copy_reseeding(ModelClass):
    env = Wrapper(ModelClass())
    seeder = Seeder(123)
    env.reseed(seeder)

    c_env = deepcopy(env)
    c_env.reseed()

    if deepcopy(env).is_online():
        traj1 = get_env_trajectory(env, 500)
        traj2 = get_env_trajectory(c_env, 500)
        assert not compare_trajectories(traj1, traj2)


@pytest.mark.parametrize("ModelClass", classes)
def test_double_wrapper_copy_reseeding(ModelClass):
    env = Wrapper(Wrapper(ModelClass()))
    seeder = Seeder(123)
    env.reseed(seeder)

    c_env = deepcopy(env)
    c_env.reseed()

    if deepcopy(env).is_online():
        traj1 = get_env_trajectory(env, 500)
        traj2 = get_env_trajectory(c_env, 500)
        assert not compare_trajectories(traj1, traj2)


def test_gym_copy_reseeding():
    seeder = Seeder(123)
    if _GYM_INSTALLED:
        gym_env = gym.make('Acrobot-v1')
        env = Wrapper(gym_env)
        env.reseed(seeder)

        c_env = deepcopy(env)
        c_env.reseed()

        if deepcopy(env).is_online():
            traj1 = get_env_trajectory(env, 500)
            traj2 = get_env_trajectory(c_env, 500)
            assert not compare_trajectories(traj1, traj2)


def test_gym_copy_reseeding_2():
    seeder = Seeder(123)
    if _GYM_INSTALLED:
        gym_env = gym.make('Acrobot-v1')
        # nested wrapping
        env = RescaleRewardWrapper(Wrapper(Wrapper(gym_env)), (0, 1))
        env.reseed(seeder)

        c_env = deepcopy(env)
        c_env.reseed()

        if deepcopy(env).is_online():
            traj1 = get_env_trajectory(env, 500)
            traj2 = get_env_trajectory(c_env, 500)
            assert not compare_trajectories(traj1, traj2)
