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

    def index(tr):
        return -tr.n_pulls if tr.t < m * A else tr.mu_hats

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

    def index(tr):
        return tr.mu_hats + sigma * np.sqrt(2 * np.log(1 / delta(tr.t)) / tr.n_pulls)

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

    Return
    ------
    Callable
        MOSS Index for sigma-sub-Gaussian distributions.

    References
    ----------
    .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
            Cambridge University Press, 2020.
    """

    def index(tr):
        return tr.mu_hats + sigma * np.sqrt(
            4 / tr.n_pulls * np.maximum(0, np.log(T / (A * tr.n_pulls)))
        )

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


def makeEXP3Index():
    """
    EXP3 index for distributions in [0, 1], see Chapters 11 in [1] and [2].

    Return
    ------
    Callable
        MOSS Index for sigma-sub-Gaussian distributions.

    References
    ----------
    .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
            Cambridge University Press, 2020.

    .. [2] Seldin, Yevgeny, et al. Evaluation and analysis of the
            performance of the EXP3 algorithm in stochastic environments.
            European Workshop on Reinforcement Learning. PMLR, 2013.
    """

    def prob(tr):
        eta = np.minimum(
            np.sqrt(np.log(tr.n_arms) / (tr.n_arms * (tr.t + 1))),
            1 / tr.n_arms,
        )
        w = np.exp(eta * tr.iw_S_hats)
        w /= w.sum()
        return (1 - tr.n_arms * eta) * w + eta * np.ones(tr.n_arms)

    return prob
