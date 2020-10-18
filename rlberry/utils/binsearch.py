import numpy as np 
from numba import jit


@jit(nopython=True)
def binary_search_1d(x, bins_1d, left_idx=0, right_idx=-1):
    """
    1-dimensional binary search

    Parameters
    -----------
    x : double
        value to be searched
    bins_1d :  numpy.ndarray or tuple
        1d numpy array or tuple representing the bins

    Returns
    --------
        Index of x in the partition defined by the bins. Returns -1 if x is not in the partition.
    """
    if(right_idx == -1):
        right_idx = len(bins_1d) - 1
    if(right_idx > left_idx):
        mid = left_idx + (right_idx-left_idx) // 2
        if(bins_1d[mid] <= x and bins_1d[mid+1]>x):
            return mid 
        if (x >= bins_1d[mid+1]):
            return binary_search_1d(x, bins_1d, mid+1, right_idx)
        if (x < bins_1d[mid]):
            return binary_search_1d(x, bins_1d, left_idx, mid)
    return -1

def binary_search_nd(x_vec, bins):
    """
    n-dimensional binary search
    
    Parameters
    -----------
    x_vec : numpy.ndarray
        numpy 1d array to be searched in the bins
    bins : list 
        list of numpy 1d array, bins[d] = bins of the d-th dimension
    
    
    Returns
    --------
    index (int) corresponding to the position of x in the partition defined by the bins.
    """
    dim = len(bins)
    flat_index = 0
    aux = 1
    assert dim == len(x_vec), "dimension mismatch in binary_search_nd()"
    for dd in range(dim):
        index_dd = binary_search_1d(x_vec[dd], bins[dd])
        assert index_dd != -1, "error in binary_search_nd()"
        flat_index += aux*index_dd 
        aux *=  (len(bins[dd]) - 1)
    return flat_index

def unravel_index_uniform_bin(flat_index, dim, n_per_dim):
    index = []
    aux_index = flat_index
    for dd in range(dim):
        index.append(aux_index %  n_per_dim)
        aux_index =  aux_index // n_per_dim 
    return tuple(index)


if __name__=="__main__":
    bins    = [ (0, 1, 2, 3, 4),
                (0, 1, 2, 3, 4)]
    x       = [3.9, 3.5]
    index   = binary_search_nd(x, bins)
    print(index)