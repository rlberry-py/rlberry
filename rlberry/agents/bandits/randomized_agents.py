import numpy as np
from rlberry.agents.bandits import BanditWithSimplePolicy
import logging

logger = logging.getLogger(__name__)


class RandomizedAgent(BanditWithSimplePolicy):
    """
    Agent for bandit environment using randomized policy like EXP3.

    Parameters
    -----------
    env : rlberry bandit environment
        See :class:`~rlberry.envs.bandits.Bandit`.

    index_function : callable or None, default = None
        Compute the index for an arm using the past rewards and sampling
        probability on this arm and the current time t.
        If None, use loss-based importance weighted estimator.

    prob_function : callable or None, default = None
        Compute the sampling probability for an arm using its index.
        If None, EXP3 softmax probabilities.
        References: Seldin, Yevgeny, et al. "Evaluation and analysis of the
        performance of the EXP3 algorithm in stochastic environments.".
        European Workshop on Reinforcement Learning. PMLR, 2013.

    Examples
    --------
    >>> from rlberry.agents.bandits import IndexAgent
    >>> import numpy as np
    >>> class EXP3Agent(RandomizedAgent):
    >>>     name = "EXP3"
    >>>     def __init__(self, env, **kwargs):
    >>>         def index(r, p, t):
    >>>             return np.sum(1 - (1 - r) / p)
    >>>
    >>>         def prob(indices, t):
    >>>             eta = np.minimum(
    >>>                 np.sqrt(np.log(self.n_arms) / (self.n_arms * (t + 1))), 1 / self.n_arms
    >>>             )
    >>>             w = np.exp(eta * indices)
    >>>             w /= w.sum()
    >>>             return (1 - self.n_arms * eta) * w + eta * np.ones(self.n_arms)
    >>>
    >>>         RandomizedAgent.__init__(self, env, index, prob, **kwargs)

    """

    name = "RandomizedAgent"

    def __init__(self, env, index_function=None, prob_function=None, **kwargs):
        BanditWithSimplePolicy.__init__(self, env, **kwargs)
        if index_function is None:

            def index_function(r, p, t):
                return np.sum(1 - (1 - r) / p)

            self.index_function = index_function
        else:
            self.index_function = index_function

        if prob_function is None:

            def prob_function(indices, t):
                eta = np.minimum(
                    np.sqrt(np.log(self.n_arms) / (self.n_arms * (t + 1))),
                    1 / self.n_arms,
                )
                w = np.exp(eta * indices)
                w /= w.sum()
                return (1 - self.n_arms * eta) * w + eta * np.ones(self.n_arms)

            self.prob_function = prob_function
        else:
            self.prob_function = prob_function
        self.total_time = 0

    def fit(self, budget=None, **kwargs):
        horizon = budget
        rewards = np.zeros(horizon)
        actions = np.ones(horizon) * np.nan
        probs = np.ones((horizon, self.n_arms)) * np.nan

        indices = np.zeros(self.n_arms)
        for ep in range(horizon):
            probs[ep] = self.prob_function(indices, ep)
            action = self.seeder.rng.choice(self.arms, p=probs[ep])
            indices = self.get_indices(rewards, actions, probs, ep)
            self.total_time += 1
            _, reward, _, _ = self.env.step(action)
            rewards[ep] = reward
            actions[ep] = action

        self.optimal_action = np.argmax(probs)
        info = {"episode_reward": np.sum(rewards)}
        return info

    def get_indices(self, rewards, actions, probs, ep):
        """
        Return the indices of each arm.

        Parameters
        ----------

        rewards : array, shape (n_iterations,)
            list of rewards until now

        actions : array, shape (n_iterations,)
            list of actions until now

        probs : array, shape (self.n_arms,)
            list of sampling probabilities for each arm

        ep: int
            current iteration/epoch

        """
        indices = np.zeros(self.n_arms)
        for a in self.arms:
            indices[a] = self.index_function(
                rewards[actions == a], probs[actions == a, a], ep
            )
        return indices
