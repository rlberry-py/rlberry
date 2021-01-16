from rlberry.utils.jit_setup import numba_jit


@numba_jit
def update_value_and_get_action(state,
                                hh,
                                V,
                                R_hat,
                                P_hat,
                                B_sa,
                                gamma,
                                v_max):
    """
    state : int
    hh : int
    V : np.ndarray
        shape (H, S)
    R_hat : np.ndarray
        shape (S, A)
    P_hat : np.ndarray
        shape (S, A, S)
    B_sa : np.ndarray
        shape (S, A)
    gamma : double
    v_max : np.ndarray
        shape (H,)
    """
    H = V.shape[0]
    S, A = R_hat.shape[-2:]
    best_action = 0
    max_val = 0
    previous_value = V[hh, state]

    for aa in range(A):
        q_aa = R_hat[state, aa] + B_sa[state, aa]

        if hh < H-1:
            for sn in range(S):
                q_aa += gamma*P_hat[state, aa, sn]*V[hh+1, sn]

        if aa == 0 or q_aa > max_val:
            max_val = q_aa
            best_action = aa

    V[hh, state] = max_val
    V[hh, state] = min(v_max[hh], V[hh, state])
    V[hh, state] = min(previous_value, V[hh, state])

    return best_action


@numba_jit
def update_value_and_get_action_sd(state,
                                   hh,
                                   V,
                                   R_hat,
                                   P_hat,
                                   B_sa,
                                   gamma,
                                   v_max):
    """
    state : int
    hh : int
    V : np.ndarray
        shape (H, S)
    R_hat : np.ndarray
        shape (H, S, A)
    P_hat : np.ndarray
        shape (H, S, A, S)
    B_sa : np.ndarray
        shape (S, A)
    gamma : double
    v_max : np.ndarray
        shape (H,)
    """
    H = V.shape[0]
    S, A = R_hat.shape[-2:]
    best_action = 0
    max_val = 0
    previous_value = V[hh, state]

    for aa in range(A):
        q_aa = R_hat[hh, state, aa] + B_sa[hh, state, aa]

        if hh < H-1:
            for sn in range(S):
                q_aa += gamma*P_hat[hh, state, aa, sn]*V[hh+1, sn]

        if aa == 0 or q_aa > max_val:
            max_val = q_aa
            best_action = aa

    V[hh, state] = max_val
    V[hh, state] = min(v_max[hh], V[hh, state])
    V[hh, state] = min(previous_value, V[hh, state])

    return best_action
