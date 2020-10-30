import numpy as np 
import pytest
import rlberry.seeding as seeding
import rlberry.spaces as spaces

from rlberry.envs.classic_control import MountainCar, Acrobot
from rlberry.envs.finite import Chain
from rlberry.envs.finite import GridWorld
from rlberry.envs.interface import OnlineModel, GenerativeModel
from rlberry.envs.toy_exploration import PBall2D
from rlberry.envs.toy_exploration import SimplePBallND

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
    for ii in range(horizon):
        states.append(ss)
        ss, _, _, _ = env.step(env.action_space.sample())
    return states

def compare_trajectories(traj1, traj2):
    for ss1, ss2 in zip(traj1, traj2):
        if not np.array_equal(ss1, ss2):
            return False 
    return True

@pytest.mark.parametrize("ModelClass", classes)
def test_env_seeding(ModelClass):
    
    seeding.set_global_seed(123)
    env1 = ModelClass()

    seeding.set_global_seed(456)
    env2 = ModelClass()

    seeding.set_global_seed(123)
    env3 = ModelClass()

    if isinstance(env1, OnlineModel):
        traj1 = get_env_trajectory(env1, 500)
        traj2 = get_env_trajectory(env2, 500)
        traj3 = get_env_trajectory(env3, 500)

        assert not compare_trajectories(traj1, traj2)
        assert compare_trajectories(traj1, traj3)


        