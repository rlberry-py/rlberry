import numpy as np
import pytest

from rlberry.spaces import Box
from rlberry.spaces import Discrete


@pytest.mark.parametrize("n", list(range(10)))
def test_discrete_space(n):
    sp = Discrete(n)
    for ii in range(n):
        assert sp.contains(ii)

    for ii in range(2 * n):
        assert sp.contains(sp.sample())


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
def test_box_space_case_1(low, high, dim):
    sp = Box(low, high, dim)
    for ii in range(2 ** dim):
        assert (sp.contains(sp.sample()))


@pytest.mark.parametrize("low, high",
                         [
                             (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
                             (np.array([-10.0, -10.0, -10.0]), np.array([10.0, 10.0, 10.0])),
                             (np.array([-10.0, -10.0, -10.0]), np.array([10.0, 10.0, np.inf])),
                             (np.array([-np.inf, -10.0, -10.0]), np.array([10.0, 10.0, np.inf])),
                             (np.array([-np.inf, -10.0, -10.0]), np.array([np.inf, 10.0, np.inf]))
                         ])
def test_box_space_case_2(low, high):
    sp = Box(low, high)
    for ii in range(2 ** sp.dim):
        assert (sp.contains(sp.sample()))
