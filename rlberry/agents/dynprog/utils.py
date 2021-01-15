import numpy as np
from rlberry.utils.jit_setup import numba_jit


@numba_jit
def backward_induction(R, P, horizon, gamma=1.0, vmax=np.inf):
    """Backward induction to compute Q and V functions in the finite horizon
    setting.

    Parameters
    ----------
    R : numpy.ndarray
        array of shape (S, A) contaning the rewards, where S is the number
        of states and A is the number of actions
    P : numpy.ndarray
        array of shape (S, A, S) such that P[s,a,ns] is the probability of
        arriving at ns by taking action a in state s.
    horizon : int
        problem horizon
    gamma : double, default: 1.0
        discount factor
    vmax : double, default: np.inf
        maximum possible value in V

    Returns
    --------
    tuple (Q, V) containing the Q and V functions, of shapes (horizon, S, A)
    and (horizon, S), respectively.
    """
    S, A = R.shape
    V = np.zeros((horizon, S))
    Q = np.zeros((horizon, S, A))
    for hh in range(horizon - 1, -1, -1):
        for ss in range(S):
            max_q = -np.inf
            for aa in range(A):
                q_aa = R[ss, aa]
                if hh < horizon - 1:
                    # not using .dot instead of loop to avoid scipy dependency
                    # (numba seems to require scipy for linear
                    # algebra operations in numpy)
                    for ns in range(S):
                        q_aa += gamma * P[ss, aa, ns] * V[hh + 1, ns]
                if q_aa > max_q:
                    max_q = q_aa
                Q[hh, ss, aa] = q_aa
            V[hh, ss] = max_q
            if V[hh, ss] > vmax:
                V[hh, ss] = vmax
    return Q, V


@numba_jit
def backward_induction_in_place(Q, V, R, P, horizon, gamma=1.0, vmax=np.inf):
    """
    Backward induction to compute Q and V functions in
    the finite horizon setting.
    Takes as input the arrays where to store Q and V.

    Parameters
    ----------
    Q:  numpy.ndarray
        array of shape (horizon, S, A) where to store the Q function
    V:  numpy.ndarray
        array of shape (horizon, S) where to store the V function
    R : numpy.ndarray
        array of shape (S, A) contaning the rewards, where S is the number
        of states and A is the number of actions
    P : numpy.ndarray
        array of shape (S, A, S) such that P[s,a,ns] is the probability of
        arriving at ns by taking action a in state s.
    horizon : int
        problem horizon
    gamma : double
        discount factor, default = 1.0
    vmax : double
        maximum possible value in V
        default = np.inf
    """
    S, A = R.shape
    for hh in range(horizon - 1, -1, -1):
        for ss in range(S):
            max_q = -np.inf
            for aa in range(A):
                q_aa = R[ss, aa]
                if hh < horizon - 1:
                    # not using .dot instead of loop to avoid scipy dependency
                    # (numba seems to require scipy for linear algebra
                    #  operations in numpy)
                    for ns in range(S):
                        q_aa += gamma * P[ss, aa, ns] * V[hh + 1, ns]
                if q_aa > max_q:
                    max_q = q_aa
                Q[hh, ss, aa] = q_aa
            V[hh, ss] = max_q
            if V[hh, ss] > vmax:
                V[hh, ss] = vmax


@numba_jit
def backward_induction_sd(Q, V, R, P, gamma=1.0, vmax=np.inf):
    """
    In-place implementation of backward induction to compute Q and V functions
    in the finite horizon setting.

    Assumes R and P are stage-dependent.

    Parameters
    ----------
    Q:  numpy.ndarray
        array of shape (H, S, A) where to store the Q function
    V:  numpy.ndarray
        array of shape (H, S) where to store the V function
    R : numpy.ndarray
        array of shape (H, S, A) contaning the rewards, where S is the number
        of states and A is the number of actions
    P : numpy.ndarray
        array of shape (H, S, A, S) such that P[h, s, a, ns] is the probability of
        arriving at ns by taking action a in state s at stage h.
    gamma : double, default: 1.0
        discount factor
    vmax : double, default: np.inf
        maximum possible value in V

    """
    H, S, A = R.shape
    for hh in range(H - 1, -1, -1):
        for ss in range(S):
            max_q = -np.inf
            for aa in range(A):
                q_aa = R[hh, ss, aa]
                if hh < H - 1:
                    # not using .dot instead of loop to avoid scipy dependency
                    # (numba seems to require scipy for linear
                    # algebra operations in numpy)
                    for ns in range(S):
                        q_aa += gamma * P[hh, ss, aa, ns] * V[hh + 1, ns]
                if q_aa > max_q:
                    max_q = q_aa
                Q[hh, ss, aa] = q_aa
            V[hh, ss] = max_q
            # clip V
            if V[hh, ss] > vmax:
                V[hh, ss] = vmax

@numba_jit
def value_iteration(R, P, gamma, epsilon=1e-6):
    """
    Value iteration for discounted problems.

    Parameters
    ----------
    R : numpy.ndarray
        array of shape (S, A) contaning the rewards, where S is the number
        of states and A is the number of actions
    P : numpy.ndarray
        array of shape (S, A, S) such that P[s,a,ns] is the probability of
        arriving at ns by taking action a in state s.
    gamma : double
        discount factor
    epsilon : double
        precision

    Returns
    --------
    tuple (Q, V, n_it) containing the epsilon-optimal Q and V functions,
    of shapes (S, A) and (S,), respectively, and n_it, the number of iterations
    """
    S, A = R.shape
    Q = np.zeros((S, A))
    Q_aux = np.full((S, A), np.inf)
    n_it = 0
    while np.abs(Q - Q_aux).max() > epsilon:
        Q_aux = Q
        Q = bellman_operator(Q, R, P, gamma)
        n_it += 1
    V = np.zeros(S)
    # numba does not support np.max(Q, axis=1)
    for ss in range(S):
        V[ss] = Q[ss, :].max()
    return Q, V, n_it


@numba_jit
def bellman_operator(Q, R, P, gamma):
    """
    Bellman optimality operator for Q functions

    Parameters
    ----------
    Q : numpy.ndarray
        array of shape (S, A) containing the Q function to which apply
        the operator
    R : numpy.ndarray
        array of shape (S, A) contaning the rewards, where S is the number
        of states and A is the number of actions
    P : numpy.ndarray
        array of shape (S, A, S) such that P[s,a,ns] is the probability of
        arriving at ns by taking action a in state s.
    gamma : double
        discount factor

    Returns
    --------
    TQ, array of shape (S, A) containing the result of the Bellman operator
    applied to the input Q
    """
    S, A = Q.shape
    TQ = np.zeros((S, A))
    V = np.zeros(S)
    # numba does not support np.max(Q, axis=1)
    for ss in range(S):
        V[ss] = Q[ss, :].max()
    #
    for ss in range(S):
        for aa in range(A):
            TQ[ss, aa] = R[ss, aa]
            for ns in range(S):
                TQ[ss, aa] += gamma * P[ss, aa, ns] * V[ns]
    return TQ

