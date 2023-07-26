import numpy as np
from rlberry.agents.bandits import BanditWithSimplePolicy


import rlberry

logger = rlberry.logger

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

    **kwargs: arguments
        Arguments to be passed to :class:`~rlberry.agents.bandit.BanditWithSimplePolicy`.
        In particular, one may want to pass the following parameters:
        tracker_params: dict
            Parameters for the tracker object, typically to decide what to store.
            in particular may contain a function "update", used to define additional statistics
            that have to be saved in the tracker. See :class:~rlberry.agents.bandit.BanditTracker`.

    Examples
    --------
    >>> from rlberry.agents.bandits import IndexAgent
    >>> import numpy as np
    >>> class UCBAgent(IndexAgent):
    >>>     name = "UCB"
    >>>     def __init__(self, env, **kwargs):
    >>>     def index(tr):
    >>>         return [
    >>>             tr.mu_hat(arm)
    >>>             + np.sqrt(
    >>>                 np.log(tr.t ** 2)
    >>>                 / (2 * tr.n_pulls(arm))
    >>>             )
    >>>             for arm in tr.arms
    >>>         ]
    >>>         IndexAgent.__init__(self, env, index, **kwargs)

    """

    name = "IndexAgent"

    def __init__(self, env, index_function=None, **kwargs):
        BanditWithSimplePolicy.__init__(self, env, **kwargs)
        if index_function is None:

            def index_function(tr):
                return [
                    tr.mu_hat(arm) + np.sqrt(np.log(tr.t**2) / (2 * tr.n_pulls(arm)))
                    for arm in tr.arms
                ]

        self.index_function = index_function

    def fit(self, budget=None, **kwargs):
        """
        Train the bandit using the provided environment.

        Parameters
        ----------
        budget: int
            Total number of iterations, also called horizon.
        **kwargs : Keyword Arguments
            Extra arguments. Not used for this agent.
        """
        horizon = budget
        total_reward = 0.0
        indices = np.inf * np.ones(self.n_arms)

        for ep in range(horizon):
            # Warmup: play every arm one before starting computing indices
            if ep < self.n_arms:
                action = ep
            else:
                # Compute index for each arm and play the highest one
                indices = self.index_function(self.tracker)
                action = np.argmax(indices)

            _, reward, _, _, _ = self.env.step(action)

            # Feed the played action and the resulting reward to the tracker
            self.tracker.update(action, reward)

            total_reward += reward

        # Best action in hinsight is the one with highest index
        self.optimal_action = np.argmax(indices)

        info = {"episode_reward": total_reward}
        return info
