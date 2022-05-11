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
    >>>         def prob_function(tr):
    >>>             w = np.zeros(tr.n_arms)
    >>>             for arm in tr.arms:
    >>>                 eta = np.minimum(
    >>>                     np.sqrt(
    >>>                         np.log(tr.n_arms) / (tr.n_arms * (tr.read_last_tag_value("t") + 1))
    >>>                     ),
    >>>                     1 / tr.n_arms,
    >>>                 )
    >>>                 w[arm] = np.exp(eta * tr.read_last_tag_value("iw_total_reward", arm))
    >>>             w /= w.sum()
    >>>             return (1 - tr.n_arms * eta) * w + eta * np.ones(tr.n_arms)
    >>>
    >>>         RandomizedAgent.__init__(self, env, index, prob, **kwargs)

    """

    name = "RandomizedAgent"

    def __init__(self, env, prob_function=None, **kwargs):
        BanditWithSimplePolicy.__init__(self, env, **kwargs)

        if prob_function is None:

            def prob_function(tr):
                w = np.zeros(tr.n_arms)
                for arm in tr.arms:
                    eta = np.minimum(
                        np.sqrt(
                            np.log(tr.n_arms)
                            / (tr.n_arms * (tr.read_last_tag_value("t") + 1))
                        ),
                        1 / tr.n_arms,
                    )
                    w[arm] = np.exp(
                        eta * tr.read_last_tag_value("iw_total_reward", arm)
                    )
                w /= w.sum()
                return (1 - tr.n_arms * eta) * w + eta * np.ones(tr.n_arms)

        self.prob_function = prob_function

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
                # Compute sampling probability for each arm
                # and play one at random
                probs = self.prob_function(self.tracker)
                action = self.rng.choice(self.arms, p=probs)

            _, reward, _, _ = self.env.step(action)

            # Feed the played action and the resulting reward and sampling
            # probability to the tracker.
            self.tracker.update(action, reward, {"p": probs[action]})

            total_reward += reward

        # Best action in hinsight is the one with highest sampling probability
        self.optimal_action = np.argmax(probs[:])
        info = {"episode_reward": total_reward}
        return info
