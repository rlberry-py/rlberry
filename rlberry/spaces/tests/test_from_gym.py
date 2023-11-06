import numpy as np
import pytest
import gymnasium.spaces
import rlberry.spaces
from rlberry.spaces.from_gym import convert_space_from_gym


@pytest.mark.parametrize("n", list(range(1, 10)))
def test_discrete_space(n):
    gym_sp = gymnasium.spaces.Discrete(n)
    sp = convert_space_from_gym(gym_sp)
    assert isinstance(sp, rlberry.spaces.Discrete)
    sp.reseed(123)
    for ii in range(n):
        assert sp.contains(ii)

    for ii in range(2 * n):
        assert sp.contains(sp.sample())


@pytest.mark.parametrize(
    "low, high, dim",
    [
        (1.0, 10.0, 1),
        (1.0, 10.0, 2),
        (1.0, 10.0, 4),
        (-10.0, 1.0, 1),
        (-10.0, 1.0, 2),
        (-10.0, 1.0, 4),
        (-np.inf, 1.0, 1),
        (-np.inf, 1.0, 2),
        (-np.inf, 1.0, 4),
        (1.0, np.inf, 1),
        (1.0, np.inf, 2),
        (1.0, np.inf, 4),
        (-np.inf, np.inf, 1),
        (-np.inf, np.inf, 2),
        (-np.inf, np.inf, 4),
    ],
)
def test_box_space_case_1(low, high, dim):
    shape = (dim, 1)
    gym_sp = gymnasium.spaces.Box(low, high, shape=shape)
    sp = convert_space_from_gym(gym_sp)
    assert isinstance(sp, rlberry.spaces.Box)
    sp.reseed(123)
    for _ in range(2**dim):
        assert sp.contains(sp.sample())


@pytest.mark.parametrize(
    "low, high",
    [
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        (np.array([-10.0, -10.0, -10.0]), np.array([10.0, 10.0, 10.0])),
        (np.array([-10.0, -10.0, -10.0]), np.array([10.0, 10.0, np.inf])),
        (np.array([-np.inf, -10.0, -10.0]), np.array([10.0, 10.0, np.inf])),
        (np.array([-np.inf, -10.0, -10.0]), np.array([np.inf, 10.0, np.inf])),
    ],
)
def test_box_space_case_2(low, high):
    gym_sp = gymnasium.spaces.Box(low, high, dtype=np.float64)
    sp = convert_space_from_gym(gym_sp)
    assert isinstance(sp, rlberry.spaces.Box)
    sp.reseed(123)
    if (-np.inf in low) or (np.inf in high):
        assert not sp.is_bounded()
    else:
        assert sp.is_bounded()
    for ii in range(2 ** sp.shape[0]):
        assert sp.contains(sp.sample())


def test_tuple():
    sp1 = gymnasium.spaces.Box(0.0, 1.0, shape=(3, 2))
    sp2 = gymnasium.spaces.Discrete(2)
    gym_sp = gymnasium.spaces.Tuple([sp1, sp2])
    sp = convert_space_from_gym(gym_sp)
    assert isinstance(sp, rlberry.spaces.Tuple)
    assert isinstance(sp.spaces[0], rlberry.spaces.Box)
    assert isinstance(sp.spaces[1], rlberry.spaces.Discrete)
    sp.reseed(123)
    for _ in range(10):
        assert sp.contains(sp.sample())


def test_multidiscrete():
    gym_sp = gymnasium.spaces.MultiDiscrete([5, 2, 2])
    sp = convert_space_from_gym(gym_sp)
    assert isinstance(sp, rlberry.spaces.MultiDiscrete)
    sp.reseed(123)
    for _ in range(10):
        assert sp.contains(sp.sample())


def test_multibinary():
    for n in [1, 5, [3, 4]]:
        gym_sp = gymnasium.spaces.MultiBinary(n)
        sp = convert_space_from_gym(gym_sp)
        assert isinstance(sp, rlberry.spaces.MultiBinary)
        for _ in range(10):
            assert sp.contains(sp.sample())
        sp.reseed(123)


def test_dict():
    nested_observation_space = gymnasium.spaces.Dict(
        {
            "sensors": gymnasium.spaces.Dict(
                {
                    "position": gymnasium.spaces.Box(low=-100, high=100, shape=(3,)),
                    "velocity": gymnasium.spaces.Box(low=-1, high=1, shape=(3,)),
                    "front_cam": gymnasium.spaces.Tuple(
                        (
                            gymnasium.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                            gymnasium.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                        )
                    ),
                    "rear_cam": gymnasium.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                }
            ),
            "ext_controller": gymnasium.spaces.MultiDiscrete((5, 2, 2)),
            "inner_state": gymnasium.spaces.Dict(
                {
                    "charge": gymnasium.spaces.Discrete(100),
                    "system_checks": gymnasium.spaces.MultiBinary(10),
                    "job_status": gymnasium.spaces.Dict(
                        {
                            "task": gymnasium.spaces.Discrete(5),
                            "progress": gymnasium.spaces.Box(low=0, high=100, shape=()),
                        }
                    ),
                }
            ),
        }
    )
    gym_sp = nested_observation_space
    sp = convert_space_from_gym(gym_sp)
    assert isinstance(sp, rlberry.spaces.Dict)
    for _ in range(10):
        assert sp.contains(sp.sample())
    sp.reseed(123)

    gym_sp2 = gymnasium.spaces.Dict(sp.spaces)
    sp2 = convert_space_from_gym(gym_sp2)
    assert isinstance(sp2, rlberry.spaces.Dict)
    for _ in range(10):
        assert sp.contains(sp2.sample())
    sp2.reseed(123)
