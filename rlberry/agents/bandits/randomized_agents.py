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
    >>>             eta = np.minimum(
    >>>                 np.sqrt(np.log(tr.n_arms) / (tr.n_arms * (tr.t + 1))),
    >>>                 1 / tr.n_arms,
    >>>             )
    >>>             w = np.exp(eta * tr.iw_S_hats)
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
                eta = np.minimum(
                    np.sqrt(np.log(tr.n_arms) / (tr.n_arms * (tr.t + 1))),
                    1 / tr.n_arms,
                )
                w = np.exp(eta * tr.iw_S_hats)
                w /= w.sum()
                return (1 - tr.n_arms * eta) * w + eta * np.ones(tr.n_arms)

        self.prob_function = prob_function

    def fit(self, budget=None, **kwargs):
        horizon = budget
        total_reward = 0.0

        for ep in range(horizon):
            probs = self.prob_function(self.tracker)
            action = self.seeder.rng.choice(self.arms, p=probs)
            _, reward, _, _ = self.env.step(action)
            self.tracker.update(action, reward, {"p": probs[action]})
            total_reward += reward

        self.optimal_action = np.argmax(probs[:])
        info = {"episode_reward": total_reward}
        return info
