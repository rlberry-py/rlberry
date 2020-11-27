import numpy as np
import pytest
import gym
import rlberry
from rlberry.wrappers.gym_utils import convert_space_from_gym


def convert_and_compare(sp, rlberry_space):
    sp_conv = convert_space_from_gym(sp)
    assert(isinstance(sp_conv, rlberry_space))
    sp_conv.reseed()
    for _ in range(100):
        assert (sp.contains(sp_conv.sample()))
        assert (sp_conv.contains(sp.sample()))


@pytest.mark.parametrize("n", list(range(1, 10)))
def test_discrete_space(n):
    sp = gym.spaces.Discrete(n)
    convert_and_compare(sp, rlberry.spaces.Discrete)


@pytest.mark.parametrize("low, high, dim",
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
                         ])
def test_box_space_case(low, high, dim):
    shape = (dim, 1)
    sp = gym.spaces.Box(low, high, shape=shape)
    convert_and_compare(sp, rlberry.spaces.Box)


def test_tuple():
    sp1 = gym.spaces.Box(0.0, 1.0, shape=(3, 2))
    sp2 = gym.spaces.Discrete(2)
    sp = gym.spaces.Tuple([sp1, sp2])
    convert_and_compare(sp, rlberry.spaces.Tuple)


def test_multidiscrete():
    sp = gym.spaces.MultiDiscrete([5, 2, 2])
    convert_and_compare(sp, rlberry.spaces.MultiDiscrete)


def test_multibinary():
    for n in [1, 5, [3, 4]]:
        sp = gym.spaces.MultiBinary(n)
        convert_and_compare(sp, rlberry.spaces.MultiBinary)


def test_dict():
    nested_observation_space = gym.spaces.Dict({
            'sensors':  gym.spaces.Dict({
                'position': gym.spaces.Box(low=-100, high=100, shape=(3,)),
                'velocity': gym.spaces.Box(low=-1, high=1, shape=(3,)),
                'front_cam': gym.spaces.Tuple((
                    gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                    gym.spaces.Box(low=0, high=1, shape=(10, 10, 3))
                )),
                'rear_cam': gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
            }),
            'ext_controller': gym.spaces.MultiDiscrete((5, 2, 2)),
            'inner_state': gym.spaces.Dict({
                'charge': gym.spaces.Discrete(100),
                'system_checks': gym.spaces.MultiBinary(10),
                'job_status': gym.spaces.Dict({
                    'task': gym.spaces.Discrete(5),
                    'progress': gym.spaces.Box(low=0, high=100, shape=()),
                })
            })
        })
    sp = nested_observation_space
    convert_and_compare(sp, rlberry.spaces.Dict)
