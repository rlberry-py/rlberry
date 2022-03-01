import numpy as np


def makeETCIndex(A=2, m=1):
    """
    Explore-Then-Commit index, see Chapter 6 in [1].

    Parameters
    ----------
    A: int
        Number of arms.

    m : int, default: 1
        Number of exploration pulls per arm.

    Return
    ------
    Callable
        ETC Index.

    References
    ----------
    .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
            Cambridge University Press, 2020.
    """

    def index(r, t):
        return -len(r) if t < m * A else np.mean(r, axis=0)

    return index


def makeSubgaussianUCBIndex(
    sigma=1.0, delta=lambda t: 1 / (1 + (t + 1) * np.log(t + 1) ** 2)
):
    """
    UCB index for sub-Gaussian distributions, see Chapters 7 & 8 in [1].

    Parameters
    ----------
    sigma : float, default: 1.0
        Sub-Gaussian parameter.

    delta: Callable,
        Confidence level. Default is tuned to have asymptotically optimal
        regret, see Chapter 8 in [1].

    Return
    ------
    Callable
        UCB Index for sigma-sub-Gaussian distributions.

    References
    ----------
    .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
            Cambridge University Press, 2020.
    """

    def index(r, t):
        return np.mean(r) + sigma * np.sqrt(2 * np.log(1 / delta(t)) / len(r))

    return index


def makeBoundedUCBIndex(
    upper_bound=1.0,
    lower_bound=0.0,
    delta=lambda t: 1 / (1 + (t + 1) * np.log(t + 1) ** 2),
):
    """
    UCB index for bounded distributions, see Chapters 7 & 8 in [1].
    By Hoeffding's lemma, such distributions are sigma-sub-Gaussian with
        sigma = (upper_bound - lower_bound) / 2.

    Parameters
    ----------
    upper_bound: float, default: 1.0
        Upper bound on the rewards.

    lower_bound: float, default: 0.0
        Lower bound on the rewards.

    delta: Callable,
        Confidence level. Default is tuned to have asymptotically optimal
        regret, see Chapter 8 in [1].

    Return
    ------
    Callable
        UCB Index for bounded distributions.

    References
    ----------
    .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
            Cambridge University Press, 2020.
    """
    return makeSubgaussianUCBIndex((upper_bound - lower_bound) / 2, delta)


def makeSubgaussianMOSSIndex(T=1, A=2, sigma=1.0):
    """
    MOSS index for sub-Gaussian distributions, see Chapters 9 in [1].

    Parameters
    ----------
    T: int
        Time horizon.

    A: int
        Number of arms.

    sigma : float, default: 1.0
        Sub-Gaussian parameter.

    delta: Callable,
        Confidence level. Default is tuned to have asymptotically optimal
        regret, see Chapter 8 in [1].

    Return
    ------
    Callable
        MOSS Index for sigma-sub-Gaussian distributions.

    References
    ----------
    .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
            Cambridge University Press, 2020.
    """

    def index(r, t):
        Na = len(r)
        return np.mean(r) + sigma * np.sqrt(4 / Na * max(0, np.log(T / (A * Na))))

    return index


def makeBoundedMOSSIndex(T=1, A=2, upper_bound=1.0, lower_bound=0.0):
    """
    MOSS index for bounded distributions, see Chapters 9 in [1].
    By Hoeffding's lemma, such distributions are sigma-sub-Gaussian with
        sigma = (upper_bound - lower_bound) / 2.

    Parameters
    ----------
    T: int
        Time horizon.

    A: int
        Number of arms.

    upper_bound: float, default: 1.0
        Upper bound on the rewards.

    lower_bound: float, default: 0.0
        Lower bound on the rewards.

    Return
    ------
    Callable
        MOSS Index for bounded distributions.

    References
    ----------
    .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
            Cambridge University Press, 2020.
    """
    return makeSubgaussianMOSSIndex(T, A, (upper_bound - lower_bound) / 2)
