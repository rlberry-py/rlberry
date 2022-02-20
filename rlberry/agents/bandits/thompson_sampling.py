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

    def __init__(self, env, prior, prior_params=None, **kwargs):
        BanditWithSimplePolicy.__init__(self, env, **kwargs)
        self.n_arms = self.env.action_space.n
        self.prior = prior
        self.prior_params = prior_params
        self.total_time = 0

    def fit(self, budget=None, **kwargs):
        if self.prior not in ["gaussian", "beta"]:
            raise ValueError(
                'This prior is not implemented yet. Valid prior are "gaussian" and "beta"'
            )
        if self.prior == "gaussian":
            # Initialize with all means equal to 0
            if self.prior_params is None:
                self.prior_params_ = (np.zeros(self.n_arms), np.ones(self.n_arms))
            else:
                self.prior_params_ = self.prior_params
        else:
            # Initialize with uniform prior
            self.prior_params_ = (np.ones(self.n_arms), np.ones(self.n_arms))

        horizon = budget

        total_reward = 0
        Na = np.zeros(self.n_arms)  # number of pulls of each arm.

        for ep in range(horizon):
            # sample according to prior
            sample_mu = self.sample_prior()
            action = np.argmax(sample_mu)
            _, reward, _, _ = self.env.step(action)
            self.update_prior(action, reward, Na)
            total_reward += reward
            Na[action] += 1

        self.optimal_action = self.get_optimal_action()
        info = {"episode_reward": total_reward}
        return info

    def sample_prior(self):
        sample = np.zeros(self.n_arms)
        if self.prior == "gaussian":
            for a in range(self.n_arms):
                sample[a] = self.rng.normal(
                    self.prior_params_[0][a], self.prior_params_[1][a]
                )
        else:
            for a in range(self.n_arms):
                sample[a] = self.rng.beta(
                    self.prior_params_[0][a], self.prior_params_[1][a]
                )
        return sample

    def update_prior(self, action, reward, Na):
        if self.prior == "gaussian":
            self.prior_params_[0][action] = Na[action] / (
                Na[action] + 1
            ) * self.prior_params_[0][action] + reward / (Na[action] + 1)
            self.prior_params_[1][action] = self.prior_params_[1][action] * np.sqrt(
                Na[action] / (Na[action] + 1)
            )
        else:
            self.prior_params_[0][action] = self.prior_params_[0][action] + reward
            self.prior_params_[1][action] = self.prior_params_[1][action] + 1 - reward

    def get_optimal_action(self):
        if self.prior == "gaussian":
            return np.argmax(self.prior_params_[0])
        else:
            return np.argmax(
                self.prior_params_[0] / (self.prior_params_[0] + self.prior_params_[1])
            )
