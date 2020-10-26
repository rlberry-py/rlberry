import numpy as np
import pytest

from rlberry.utils.binsearch import binary_search_nd
from rlberry.utils.binsearch import unravel_index_uniform_bin


def test_binary_search_nd():
    bin1 = np.array([0.0, 1.0, 2.0, 3.0])  # 3 intervals
    bin2 = np.array([1.0, 2.0, 3.0, 4.0])  # 3 intervals
    bin3 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # 4 intervals

    bins = [bin1, bin2, bin3]

    vec1 = np.array([0.0, 1.0, 2.0])
    vec2 = np.array([2.9, 3.9, 5.9])
    vec3 = np.array([1.5, 2.5, 2.5])
    vec4 = np.array([1.5, 2.5, 2.5])

    # index = i + Ni * j + Ni * Nj * k
    assert binary_search_nd(vec1, bins) == 0
    assert binary_search_nd(vec2, bins) == 2 + 3 * 2 + 3 * 3 * 3
    assert binary_search_nd(vec3, bins) == 1 + 3 * 1 + 3 * 3 * 0


@pytest.mark.parametrize("i, j, k, N",
                         [
                             (0, 0, 0, 5),
                             (0, 1, 2, 5),
                             (4, 3, 2, 5),
                             (4, 4, 4, 5)
                         ])
def test_unravel_index_uniform_bin(i, j, k, N):
    # index = i + N * j + N * N * k
    dim = 3
    flat_index = i + N * j + N * N * k
    assert (i, j, k) == unravel_index_uniform_bin(flat_index, dim, N)
