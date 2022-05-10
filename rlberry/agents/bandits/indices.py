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
        ETC index.

    Dict
        Extra parameters for the BanditTracker object.
        By default the tracker stores the number of pulls and the
        empirical average reward for each arm. If you want it to store
        all rewards for instance, return {'store_rewards': True}.

    References
    ----------
    .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
            Cambridge University Press, 2020.
    """

    def index(tr):
        # return -tr.n_pulls if tr.t < m * A else tr.mu_hats
        return [
            -tr.read_last_tag_value("n_pulls", arm)
            if tr.read_last_tag_value("t") < m * A
            else tr.read_last_tag_value("mu_hat", arm)
            for arm in tr.arms
        ]

    return index, {}


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
        UCB index for sigma-sub-Gaussian distributions.

    Dict
        Extra parameters for the BanditTracker object.
        By default the tracker stores the number of pulls and the
        empirical average reward for each arm. If you want it to store
        all rewards for instance, return {'store_rewards': True}.

    References
    ----------
    .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
            Cambridge University Press, 2020.
    """

    def index(tr):

        return [
            tr.read_last_tag_value("mu_hat", arm)
            + sigma
            * np.sqrt(
                2
                * np.log(1 / delta(tr.read_last_tag_value("t")))
                / tr.read_last_tag_value("n_pulls", arm)
            )
            for arm in tr.arms
        ]

    return index, {}


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
        UCB index for bounded distributions.

    Dict
        Extra parameters for the BanditTracker object.
        By default the tracker stores the number of pulls and the
        empirical average reward for each arm. If you want it to store
        all rewards for instance, return {'store_rewards': True}.

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
        MOSS index for sigma-sub-Gaussian distributions.

    Dict
        Extra parameters for the BanditTracker object.
        By default the tracker stores the number of pulls and the
        empirical average reward for each arm. If you want it to store
        all rewards for instance, return {'store_rewards': True}.

    References
    ----------
    .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
            Cambridge University Press, 2020.
    """

    def index(tr):
        return [
            tr.read_last_tag_value("mu_hat", arm)
            + sigma
            * np.sqrt(
                4
                / tr.read_last_tag_value("n_pulls", arm)
                * np.maximum(
                    0, np.log(T / (A * tr.read_last_tag_value("n_pulls", arm)))
                )
            )
            for arm in tr.arms
        ]

    return index, {}


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
        MOSS index for bounded distributions.

    Dict
        Extra parameters for the BanditTracker object.
        By default the tracker stores the number of pulls and the
        empirical average reward for each arm. If you want it to store
        all rewards for instance, return {'store_rewards': True}.

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
        EXP3 index for [0, 1] distributions.

    Dict
        Extra parameters for the BanditTracker object.
        By default the tracker stores the number of pulls and the
        empirical average reward for each arm. If you want it to store
        all rewards for instance, return {'store_rewards': True}.

    References
    ----------
    .. [1] Lattimore, Tor, and Csaba Szepesvári. Bandit algorithms.
            Cambridge University Press, 2020.

    .. [2] Seldin, Yevgeny, et al. Evaluation and analysis of the
            performance of the EXP3 algorithm in stochastic environments.
            European Workshop on Reinforcement Learning. PMLR, 2013.
    """

    def prob(tr):
        w = np.zeros(tr.n_arms)
        for arm in tr.arms:
            eta = np.minimum(
                np.sqrt(
                    np.log(tr.n_arms) / (tr.n_arms * (tr.read_last_tag_value("t") + 1))
                ),
                1 / tr.n_arms,
            )
            w[arm] = np.exp(eta * tr.read_last_tag_value("iw_total_reward", arm))
        w /= w.sum()
        return (1 - tr.n_arms * eta) * w + eta * np.ones(tr.n_arms)

    return prob, {"do_iwr": True}


def makeBoundedIMEDIndex(upper_bound=1.0):
    """
    IMED index for semi-bounded distributions, see [1].

    Parameters
    ----------
    upper_bound: float, default: 1.0
        Upper bound on the rewards.

    Return
    ------
    Callable
        IMED index for sigma-sub-Gaussian distributions.

    Dict
        Extra parameters for the BanditTracker object.
        By default the tracker stores the number of pulls and the
        empirical average reward for each arm. If you want it to store
        all rewards for instance, return {'store_rewards': True}.

    References
    ----------
    .. [1] Honda, Junya, and Akimichi Takemura. Non-asymptotic analysis of
            a new bandit algorithm for semi-bounded rewards.
            J. Mach. Learn. Res. 16 (2015): 3721-3756.
    """
    from scipy.optimize import minimize_scalar

    def index(tr):
        mu_hat_star = np.max([tr.read_last_tag_value("mu_hat", arm) for arm in tr.arms])
        indices = np.zeros(tr.n_arms)
        for arm in tr.arms:
            X = np.array(tr.read_tag_value("reward", arm))

            def dual(u):
                return -np.mean(np.log(1 - (X - mu_hat_star) * u))

            eps = 1e-12
            ret = minimize_scalar(
                dual,
                method="bounded",
                bounds=(eps, 1.0 / (upper_bound - mu_hat_star + eps)),
            )
            if ret.success:
                kinf = -ret.fun
            else:
                # if not successful, just make this arm ineligible this turn
                kinf = np.inf

            indices[arm] = -kinf * len(X) - np.log(len(X))
        return indices

    return index, {"store_rewards": True}


def makeBoundedNPTSIndex(upper_bound=1.0):
    """
    NPTS index for bounded distributions, see [1].

    Parameters
    ----------
    upper_bound: float, default: 1.0
        Upper bound on the rewards.


    Return
    ------
    Callable
        NPTS index for sigma-sub-Gaussian distributions.

    Dict
        Extra parameters for the BanditTracker object.
        By default the tracker stores the number of pulls and the
        empirical average reward for each arm. If you want it to store
        all rewards for instance, return {'store_rewards': True}.

    References
    ----------
    .. [1] Riou, Charles, and Junya Honda. Bandit algorithms based on
            thompson sampling for bounded reward distributions.
            Algorithmic Learning Theory. PMLR, 2020.

    """

    def index(tr):
        indices = np.zeros(tr.n_arms)
        for arm in tr.arms:
            X = np.array(tr.read_tag_value("reward", arm))
            w = tr.seeder.rng.dirichlet(np.ones(len(X) + 1))
            indices[arm] = w[:-1] @ X + upper_bound * w[-1]
        return indices

    return index, {"store_rewards": True}
