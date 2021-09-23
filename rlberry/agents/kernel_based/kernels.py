import numpy as np
from rlberry.utils.jit_setup import numba_jit


@numba_jit
def kernel_func(z, kernel_type):
    """
    Returns a kernel function to the real value z.

    Kernel types:

    "uniform"      : 1.0*(abs(z) <= 1)
    "triangular"   : max(0, 1 - abs(z))
    "gaussian"     : exp(-z^2/2)
    "epanechnikov" : max(0, 1-z^2)
    "quartic"      : (1-z^2)^2 *(abs(z) <= 1)
    "triweight"    : (1-z^2)^3 *(abs(z) <= 1)
    "tricube"      : (1-abs(z)^3)^3 *(abs(z) <= 1)
    "cosine"       : cos( z * (pi/2) ) *(abs(z) <= 1)
    "exp-n"        : exp(-abs(z)^n/2), for n integer

    Parameters
    ----------
    z : double
    kernel_type : string
    """
    if kernel_type == "uniform":
        return 1.0 * (np.abs(z) <= 1)
    elif kernel_type == "triangular":
        return (1.0 - np.abs(z)) * (np.abs(z) <= 1)
    elif kernel_type == "gaussian":
        return np.exp(-np.power(z, 2.0) / 2.0)
    elif kernel_type == "epanechnikov":
        return (1.0 - np.power(z, 2.0)) * (np.abs(z) <= 1)
    elif kernel_type == "quartic":
        return np.power((1.0 - np.power(z, 2.0)), 2.0) * (np.abs(z) <= 1)
    elif kernel_type == "triweight":
        return np.power((1.0 - np.power(z, 2.0)), 3.0) * (np.abs(z) <= 1)
    elif kernel_type == "tricube":
        return np.power((1.0 - np.power(np.abs(z), 3.0)), 3.0) * (np.abs(z) <= 1)
    elif kernel_type == "cosine":
        return np.cos(z * np.pi / 2) * (np.abs(z) <= 1)
    elif "exp-" in kernel_type:
        exponent = _str_to_int(kernel_type.split("-")[1])
        return np.exp(-np.power(np.abs(z), exponent) / 2.0)
    else:
        raise NotImplementedError("Unknown kernel type.")


@numba_jit
def _str_to_int(s):
    """
    Source: https://github.com/numba/numba/issues/5650#issuecomment-623511109
    """
    final_index, result = len(s) - 1, 0
    for i, v in enumerate(s):
        result += (ord(v) - 48) * (10 ** (final_index - i))
    return result
