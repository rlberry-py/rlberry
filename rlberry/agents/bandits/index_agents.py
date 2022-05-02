import numpy as np
from rlberry.agents.bandits import BanditWithSimplePolicy
import logging

logger = logging.getLogger(__name__)

# TODO : fix bug when doing several fit, the fit do not resume. Should define
#        self.rewards and self.action and resume training.


class IndexAgent(BanditWithSimplePolicy):
    """
    Agent for bandit environment using Index-based policy like UCB.

    Parameters
    -----------
    env : rlberry bandit environment
        See :class:`~rlberry.envs.bandits.Bandit`.

    index_function : callable or None, default = None
        Compute the index for an arm using the past rewards on this arm and
        the current time t. If None, use UCB bound for Bernoulli.


    Examples
    --------
    >>> from rlberry.agents.bandits import IndexAgent
    >>> import numpy as np
    >>> class UCBAgent(IndexAgent):
    >>>     name = "UCB"
    >>>     def __init__(self, env, **kwargs):
    >>>         def index_function(tr):
    >>>             return tr.mu_hats + np.sqrt(np.log(tr.t ** 2) / (2 * tr.n_pulls))
    >>>         IndexAgent.__init__(self, env, index, **kwargs)

    """

    name = "IndexAgent"

    def __init__(self, env, index_function=None, **kwargs):
        BanditWithSimplePolicy.__init__(self, env, **kwargs)
        if index_function is None:

            def index_function(tr):
                return tr.mu_hats + np.sqrt(np.log(tr.t**2) / (2 * tr.n_pulls))

        self.index_function = index_function

    def fit(self, budget=None, **kwargs):
        horizon = budget
        total_reward = 0.0
        indices = np.inf * np.ones(self.n_arms)

        for ep in range(horizon):
            if self.total_time < self.n_arms:
                action = self.total_time % self.n_arms
            else:
                indices = self.index_function(self.tracker)
                action = np.argmax(indices)
            _, reward, _, _ = self.env.step(action)
            self.tracker.update(action, reward)
            total_reward += reward

        self.optimal_action = np.argmax(indices)
        info = {"episode_reward": total_reward}
        return info
