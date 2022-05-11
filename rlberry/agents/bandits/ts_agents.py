import numpy as np
from rlberry.agents.bandits import BanditWithSimplePolicy
import logging

logger = logging.getLogger(__name__)


class TSAgent(BanditWithSimplePolicy):
    """
    Agent for bandit environment using Thompson sampling.

    Parameters
    -----------
    env : rlberry bandit environment
        See :class:`~rlberry.envs.bandits.Bandit`.

    prior : str in {"gaussian", "beta"}
        Family of priors used in Thompson sampling algorithm.

    prior_params : arary of size (2,n_actions) or None, default = None
        Only used if prior = "gaussian", means and std of the gaussian prior distributions.
        If None, use an array of all 0 and an array of all 1.
    """

    name = "TSAgent"

    def __init__(self, env, prior_info=None, **kwargs):
        BanditWithSimplePolicy.__init__(self, env, **kwargs)
        if prior_info is None:

            # Beta-Bernoulli prior by default
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
                    [
                        params[arm][0] / (params[arm][0] + params[arm][1])
                        for arm in tr.arms
                    ]
                )

            self.prior_info = {
                "params": prior_params,
                "sampler": prior_sampler,
                "optimal_action": optimal_action,
            }
        else:
            self.prior_info = prior_info

    @property
    def prior_sampler(self):
        return self.prior_info.get("sampler")

    @property
    def get_optimal_action(self):
        return self.prior_info.get("optimal_action")

    def fit(self, budget=None, **kwargs):
        """
        Train the bandit using the provided environment.

        Parameters
        ----------
        budget: int
            Total number of iterations, also called horizon.
        """
        horizon = budget

        total_reward = 0.0

        for ep in range(horizon):
            # Warmup: play every arm one before starting computing indices
            if ep < self.n_arms:
                action = ep
            else:
                # Sample from mean parameters from prior distributions
                sample_mu = self.prior_sampler(self.tracker)
                # Play the best sampled mean
                action = np.argmax(sample_mu)

            _, reward, _, _ = self.env.step(action)

            # Feed the played action and the resulting reward to the tracker
            self.tracker.update(action, reward)

            total_reward += reward

        # Best action in hinsight is the one with highest index
        self.optimal_action = self.get_optimal_action(self.tracker)

        info = {"episode_reward": total_reward}
        return info
