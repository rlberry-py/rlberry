import numpy as np
from numba import jit
from rlberry.utils.metrics import metric_lp


@jit(nopython=True)
def map_to_representative(state,
                          lp_metric,
                          representative_states,
                          n_representatives,
                          min_dist,
                          scaling,
                          accept_new_repr):
    """ Map state to representative state. """
    dist_to_closest = np.inf
    argmin = -1
    for ii in range(n_representatives):
        dist = metric_lp(state, representative_states[ii, :],
                         lp_metric,
                         scaling)
        if dist < dist_to_closest:
            dist_to_closest = dist
            argmin = ii

    max_representatives = representative_states.shape[0]
    if (dist_to_closest > min_dist) \
        and (n_representatives < max_representatives) \
            and accept_new_repr:
        new_index = n_representatives
        representative_states[new_index, :] = state
        return new_index
    return argmin
