import pytest

import numpy as np

from rlberry.spaces import Discrete 
from rlberry.spaces import Box

@pytest.mark.parametrize("n", list(range(10)))
def test_discrete_space(n):
    sp = Discrete(n)
    for ii in range(n):
        assert sp.contains(ii)

    for ii in range(2*n):
        assert sp.contains(sp.sample())

@pytest.mark.parametrize("low, high, dim", 
                         [
                           (1.0,  10.0, 1),
                           (1.0,  10.0, 2),
                           (1.0,  10.0, 4),
                           (-10.0, 1.0, 1),
                           (-10.0, 1.0, 2),
                           (-10.0, 1.0, 4),
                           (-np.inf, 1.0, 1),
                           (-np.inf, 1.0, 2),
                           (-np.inf, 1.0, 4),
                           (1.0,  np.inf, 1),
                           (1.0,  np.inf, 2),
                           (1.0,  np.inf, 4),
                           (-np.inf,  np.inf, 1),
                           (-np.inf,  np.inf, 2),
                           (-np.inf,  np.inf, 4),
                         ])
def test_box_space_case_1(low, high, dim):
    sp = Box(low, high, dim)
    for ii in range(2**dim):
        assert(sp.contains(sp.sample()))
    
