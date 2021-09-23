import numpy as np
import pytest
import rlberry.seeding as seeding

from copy import deepcopy
from rlberry.envs.classic_control import MountainCar, Acrobot, Pendulum
from rlberry.envs.finite import Chain
from rlberry.envs.finite import GridWorld
from rlberry.envs.benchmarks.grid_exploration.four_room import FourRoom
from rlberry.envs.benchmarks.grid_exploration.six_room import SixRoom
from rlberry.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry.envs.benchmarks.ball_exploration import PBall2D, SimplePBallND

classes = [
    MountainCar,
    GridWorld,
    Chain,
    PBall2D,
    SimplePBallND,
    Acrobot,
    Pendulum,
    FourRoom,
    SixRoom,
    AppleGold
]


def get_env_trajectory(env, horizon):
    states = []
    ss = env.reset()
    for ii in range(horizon):
        states.append(ss)
        ss, _, _, _ = env.step(env.action_space.sample())
    return states


def compare_trajectories(traj1, traj2):
    """
    returns true if trajectories are equal
    """
    for ss1, ss2 in zip(traj1, traj2):
        if not np.array_equal(ss1, ss2):
            return False
    return True


@pytest.mark.parametrize("ModelClass", classes)
def test_env_seeding(ModelClass):
    env1 = ModelClass()
    seeder1 = seeding.Seeder(123)
    env1.reseed(seeder1)

    env2 = ModelClass()
    seeder2 = seeder1.spawn()
    env2.reseed(seeder2)

    env3 = ModelClass()
    seeder3 = seeding.Seeder(123)
    env3.reseed(seeder3)

    env4 = ModelClass()
    seeder4 = seeding.Seeder(123)
    env4.reseed(seeder4)

    env5 = ModelClass()
    env5.reseed(seeder1)  # same seeder as env1, but different trajectories. This is expected.

    seeding.safe_reseed(env4, seeder4)

    if deepcopy(env1).is_online():
        traj1 = get_env_trajectory(env1, 500)
        traj2 = get_env_trajectory(env2, 500)
        traj3 = get_env_trajectory(env3, 500)
        traj4 = get_env_trajectory(env4, 500)
        traj5 = get_env_trajectory(env5, 500)

        assert not compare_trajectories(traj1, traj2)
        assert compare_trajectories(traj1, traj3)
        assert not compare_trajectories(traj3, traj4)
        assert not compare_trajectories(traj1, traj5)


@pytest.mark.parametrize("ModelClass", classes)
def test_copy_reseeding(ModelClass):
    seeder = seeding.Seeder(123)
    env = ModelClass()
    env.reseed(seeder)

    c_env = deepcopy(env)
    c_env.reseed()

    if deepcopy(env).is_online():
        traj1 = get_env_trajectory(env, 500)
        traj2 = get_env_trajectory(c_env, 500)
        assert not compare_trajectories(traj1, traj2)
