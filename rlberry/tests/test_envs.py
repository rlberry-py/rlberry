from rlberry.utils.check_env import check_env, check_rlberry_env
from rlberry.envs import Acrobot
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.envs.classic_control import MountainCar, SpringCartPole
from rlberry.envs.finite import Chain, GridWorld
import pytest

ALL_ENVS = [
    Acrobot,
    PBall2D,
    TwinRooms,
    AppleGold,
    NRoom,
    MountainCar,
    Chain,
    GridWorld,
    SpringCartPole,
]


@pytest.mark.parametrize("Env", ALL_ENVS)
def test_env(Env):
    check_env(Env())
    check_rlberry_env(Env())
