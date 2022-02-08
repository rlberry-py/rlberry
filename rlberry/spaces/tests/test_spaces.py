import numpy as np
import pytest

from rlberry.spaces import Box
from rlberry.spaces import Discrete
from rlberry.spaces import Tuple
from rlberry.spaces import MultiDiscrete
from rlberry.spaces import MultiBinary
from rlberry.spaces import Dict


@pytest.mark.parametrize("n", list(range(1, 10)))
def test_discrete_space(n):
    sp = Discrete(n)
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
    sp = Box(low, high, shape=shape)
    for ii in range(2**dim):
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
    sp = Box(low, high)
    if (-np.inf in low) or (np.inf in high):
        assert not sp.is_bounded()
    else:
        assert sp.is_bounded()
    for ii in range(2 ** sp.shape[0]):
        assert sp.contains(sp.sample())


def test_tuple():
    sp1 = Box(0.0, 1.0, shape=(3, 2))
    sp2 = Discrete(2)
    sp = Tuple([sp1, sp2])
    for _ in range(10):
        assert sp.contains(sp.sample())
    sp.reseed()


def test_multidiscrete():
    sp = MultiDiscrete([5, 2, 2])
    for _ in range(10):
        assert sp.contains(sp.sample())
    sp.reseed()


def test_multibinary():
    for n in [1, 5, [3, 4]]:
        sp = MultiBinary(n)
        for _ in range(10):
            assert sp.contains(sp.sample())
        sp.reseed()


def test_dict():
    nested_observation_space = Dict(
        {
            "sensors": Dict(
                {
                    "position": Box(low=-100, high=100, shape=(3,)),
                    "velocity": Box(low=-1, high=1, shape=(3,)),
                    "front_cam": Tuple(
                        (
                            Box(low=0, high=1, shape=(10, 10, 3)),
                            Box(low=0, high=1, shape=(10, 10, 3)),
                        )
                    ),
                    "rear_cam": Box(low=0, high=1, shape=(10, 10, 3)),
                }
            ),
            "ext_controller": MultiDiscrete((5, 2, 2)),
            "inner_state": Dict(
                {
                    "charge": Discrete(100),
                    "system_checks": MultiBinary(10),
                    "job_status": Dict(
                        {
                            "task": Discrete(5),
                            "progress": Box(low=0, high=100, shape=()),
                        }
                    ),
                }
            ),
        }
    )
    sp = nested_observation_space
    for _ in range(10):
        assert sp.contains(sp.sample())
    sp.reseed()

    sp2 = Dict(sp.spaces)

    for _ in range(10):
        assert sp.contains(sp2.sample())
    sp2.reseed()
