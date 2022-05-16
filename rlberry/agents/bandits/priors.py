import numpy as np


def makeBetaPrior():
    """
    Beta prior for Bernoulli bandits, see Chapter 3 in [1].

    Parameters
    ----------
        None

    Return
    ------
    Dict
        Callable
            Beta sampler.

        Callable
            Function that computes the parameters of the prior distribution
            from the bandit tracker.

        Callable
            Function that computes the optimal action from the prior distributions.

    Dict
        Extra parameters for the BanditTracker object.
        By default the tracker stores the number of pulls and the
        empirical average reward for each arm. If you want it to store
        all rewards for instance, return {'store_rewards': True}.

    References
    ----------
    .. [1] Russo, Daniel J., et al. "A tutorial on Thompson Sampling."
        Foundations and Trends in Machine Learning 11.1 (2018): 1-96.
    """

    def prior_params(tr):
        """
        The mean of a Bernoulli arm B(p) has prior distribution Beta(a, b),
        where a is the number of success + 1, b the number of failures + 1.
        """
        return [
            [
                tr.read_last_tag_value("total_reward", arm) + 1,
                tr.read_last_tag_value("n_pulls", arm)
                - tr.read_last_tag_value("total_reward", arm)
                + 1,
            ]
            for arm in tr.arms
        ]

    def prior_sampler(tr):
        """
        Beta prior.
        """
        params = prior_params(tr)
        return [tr.rng.beta(params[arm][0], params[arm][1]) for arm in tr.arms]

    def optimal_action(tr):
        """
        The mean of a Bernoulli arm B(p) has prior distribution Beta(a, b),
        where a is the number of success + 1, b the number of failures + 1.
        The expectation of p is a / (a + b), therefore the optimal arm w.r.t
        the Beta prior is the one with highest a / (a + b).
        """
        params = prior_params(tr)
        return np.argmax(
            [params[arm][0] / (params[arm][0] + params[arm][1]) for arm in tr.arms]
        )

    prior_info = {
        "params": prior_params,
        "sampler": prior_sampler,
        "optimal_action": optimal_action,
    }

    return prior_info, {}


def makeGaussianPrior(sigma: float = 1.0):
    """
    Gaussian prior for Gaussian bandits with known variance, see [1].

    Parameters
    ----------
    sigma : float, default: 1.0
        Gaussian standard deviation.

    Return
    ------
    Dict
        Callable
            Gaussian sampler.

        Callable
            Function that computes the parameters of the prior distribution
            from the bandit tracker.

        Callable
            Function that computes the optimal action from the prior distributions.

    Dict
        Extra parameters for the BanditTracker object.
        By default the tracker stores the number of pulls and the
        empirical average reward for each arm. If you want it to store
        all rewards for instance, return {'store_rewards': True}.

    References
    ----------
    .. [1] Korda, Nathaniel, Emilie Kaufmann, and Remi Munos.
        "Thompson sampling for 1-dimensional exponential family bandits."
        Advances in Neural Information Processing Systems 26 (2013).
    """

    def prior_params(tr):
        """
        The mean of a Gaussian arm N(mu, sigma^2) has prior distribution
        N(mu_hat, s^2), where mu_hat is the empirical average reward and
        s^2 = sigma^2 / n, n being the number of pulls for this arm.
        """
        return [
            [
                tr.read_last_tag_value("mu_hat", arm),
                sigma / np.sqrt(tr.read_last_tag_value("n_pulls", arm)),
            ]
            for arm in tr.arms
        ]

    def prior_sampler(tr):
        """
        Normal prior.
        """
        params = prior_params(tr)
        return [tr.rng.normal(params[arm][0], params[arm][1]) for arm in tr.arms]

    def optimal_action(tr):
        """
        The mean of a Gaussian arm N(mu, sigma^2) has prior distribution
        N(mu_hat, s^2), where mu_hat is the empirical average reward and
        s^2 = sigma^2 / n, n being the number of pulls for this arm.
        The expectation of mu is mu_hat, therefore the optimal arm w.r.t
        the Gaussian prior is the one with highest mu_hat.
        """
        params = prior_params(tr)
        return np.argmax([params[arm][0] for arm in tr.arms])

    prior_info = {
        "params": prior_params,
        "sampler": prior_sampler,
        "optimal_action": optimal_action,
    }

    return prior_info, {}
