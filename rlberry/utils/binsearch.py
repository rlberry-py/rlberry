import numpy as np


def binary_search_nd(x_vec, bins):
    """n-dimensional binary search

    Parameters
    -----------
    x_vec : numpy.ndarray
        numpy 1d array to be searched in the bins
    bins : list
        list of numpy 1d array, bins[d] = bins of the d-th dimension


    Returns
    --------
    index (int) corresponding to the position of x in the partition
    defined by the bins.
    """
    dim = len(bins)
    flat_index = 0
    aux = 1
    assert dim == len(x_vec), "dimension mismatch in binary_search_nd()"
    for dd in range(dim):
        index_dd = np.searchsorted(bins[dd], x_vec[dd], side='right') - 1
        assert index_dd != -1, "error in binary_search_nd()"
        flat_index += aux * index_dd
        aux *= (len(bins[dd]) - 1)
    return flat_index


def unravel_index_uniform_bin(flat_index, dim, n_per_dim):
    index = []
    aux_index = flat_index
    for _ in range(dim):
        index.append(aux_index % n_per_dim)
        aux_index = aux_index // n_per_dim
    return tuple(index)


if __name__ == "__main__":
    bins = [(0, 1, 2, 3, 4),
            (0, 1, 2, 3, 4)]
    x = [3.9, 3.5]
    index = binary_search_nd(x, bins)
    print(index)

