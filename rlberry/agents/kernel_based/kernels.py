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
        return np.power((1.0 - np.power(z, 2.0)), 2.0)*(np.abs(z) <= 1)
    elif kernel_type == "triweight":
        return np.power((1.0 - np.power(z, 2.0)), 3.0)*(np.abs(z) <= 1)
    elif kernel_type == "tricube":
        return np.power((1.0 - np.power(np.abs(z), 3.0)), 3.0)*(np.abs(z) <= 1)
    elif kernel_type == "cosine":
        return np.cos(z*np.pi/2)*(np.abs(z) <= 1)
    else:
        raise NotImplementedError("Unknown kernel type.")
