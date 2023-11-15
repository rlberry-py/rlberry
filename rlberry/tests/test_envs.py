from rlberry.utils.check_env import check_env, check_rlberry_env
from rlberry-research.envs import Acrobot
from rlberry-research.envs.benchmarks.ball_exploration import PBall2D
from rlberry-research.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry-research.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry-research.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry-research.envs.classic_control import MountainCar, SpringCartPole
from rlberry-research.envs.finite import Chain, GridWorld
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
