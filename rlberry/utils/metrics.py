import numpy as np
from rlberry.utils.jit_setup import numba_jit


@numba_jit
def metric_lp(x, y, p, scaling):
    """
    Returns the p-norm:  || (x-y)/scaling||_p

    Parameters
    ----------
    x : numpy.ndarray
        1d array
    y : numpy.ndarray
        1d array
    p : int
        norm parameter
    scaling : numpy.ndarray
        1d array
    """
    assert p >= 1
    assert x.ndim == 1
    assert y.ndim == 1
    assert scaling.ndim == 1

    d = len(x)
    diff = np.abs((x - y) / scaling)
    # p = infinity
    if p == np.inf:
        return diff.max()
    # p < infinity
    tmp = 0
    for ii in range(d):
        tmp += np.power(diff[ii], p)
    return np.power(tmp, 1.0 / p)
