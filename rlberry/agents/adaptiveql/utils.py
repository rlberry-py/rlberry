from rlberry.utils.jit_setup import numba_jit


@numba_jit
def bounds_contains(bounds, x):
    """
    Returns True if `x` is contained in the bounds, and False otherwise.

    Parameters
    ----------
    bounds : numpy.ndarray
        Array of shape (d, 2).
        Bounds of each dimension [ [x0, y0], [x1, y1], ..., [xd, yd] ],
        representing the following cartesian product in R^d:
        [x0, y0] X [x1, y1] X ... X [xd, yd].
    x : numpy.ndarray
        Array of shape (d,)
    """
    dim = x.shape[0]
    for dd in range(dim):
        if x[dd] < bounds[dd, 0] or x[dd] > bounds[dd, 1]:
            return False
    return True


def split_bounds(bounds, dim=0):
    """
    Split an array representing an l-infinity ball in R^d in R^d
    into a list of 2^d arrays representing the ball split.

    Parameters
    ----------
    bounds : numpy.ndarray
        Array of shape (d, 2).
        Bounds of each dimension [ [x0, y0], [x1, y1], ..., [xd, yd] ],
        representing the cartesian product in R^d:
        [x0, y0] X [x1, y1] X ... X [xd, yd].

    dim : int, default: 0
        Dimension from which to start splitting.

    Returns
    -------
    List of arrays of shape (d, 2) containing the bounds to be split.
    """
    if dim == bounds.shape[0]:
        return [bounds]
    left = bounds[dim, 0]
    right = bounds[dim, 1]
    middle = (left + right) / 2.0

    left_interval = bounds.copy()
    right_interval = bounds.copy()

    left_interval[dim, 0] = left
    left_interval[dim, 1] = middle

    right_interval[dim, 0] = middle
    right_interval[dim, 1] = right

    return split_bounds(left_interval, dim + 1) + split_bounds(right_interval, dim + 1)
