import numpy as np
from rlberry.agents.bandits import BanditWithSimplePolicy
import logging

logger = logging.getLogger(__name__)


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
    >>>         def index(r, t):
    >>>             return np.mean(r) + np.sqrt(np.log(t**2) / (2 * len(r)))
    >>>         IndexAgent.__init__(self, env, index, **kwargs)

    """

    name = "IndexAgent"

    def __init__(self, env, index_function=None, **kwargs):
        BanditWithSimplePolicy.__init__(self, env, **kwargs)
        if index_function is None:

            def index(r, t):
                return np.mean(r) + np.sqrt(np.log(t**2) / (2 * len(r)))

            self.index_function = index_function
        else:
            self.index_function = index_function
        self.total_time = 0

    def fit(self, budget=None, **kwargs):
        horizon = budget
        rewards = np.zeros(horizon)
        actions = np.ones(horizon) * np.nan

        indices = np.inf * np.ones(self.n_arms)
        for ep in range(horizon):
            if self.total_time < self.n_arms:
                action = self.total_time
            else:
                indices = self.get_indices(rewards, actions, ep)
                action = np.argmax(indices)
            self.total_time += 1
            _, reward, _, _ = self.env.step(action)
            rewards[ep] = reward
            actions[ep] = action

        self.optimal_action = np.argmax(indices)
        info = {"episode_reward": np.sum(rewards)}
        return info

    def get_indices(self, rewards, actions, ep):
        """
        Return the indices of each arm.

        Parameters
        ----------

        rewards : array, shape (n_iterations,)
            list of rewards until now

        actions : array, shape (n_iterations,)
            list of actions until now

        ep: int
            current iteration/epoch

        """
        indices = np.zeros(self.n_arms)
        for a in self.arms:
            indices[a] = self.index_function(rewards[actions == a], ep)
        return indices


class RecursiveIndexAgent(BanditWithSimplePolicy):
    """
    Agent for bandit environment using Index-based policy like UCB with
    recursive indices.

    Parameters
    -----------
    env : rlberry bandit environment

    stat_function : tuple of callable or None, default=None
        compute sufficient statistics using previous stats, Na and current action and rewards.
        If None, use the empirical mean statistic.

    index_function : callable or None, default=None
        compute the index using the stats from stat_function, the number of  pull
        of the arm and the current time.
        If None, use UCB index for bernoulli.


    Examples
    --------
    >>> from rlberry.agents.bandits import RecursiveIndexAgent
    >>> import numpy as np
    >>> class UCBAgent(RecursiveIndexAgent):
    >>>     name = "UCB Agent"
    >>>     def __init__(self, env,**kwargs):
    >>>         def stat_function(stat, Na, action, reward):
    >>>             # The statistic is the empirical mean. We compute it recursively.
    >>>             if stat is None:
    >>>                 stat = np.zeros(len(Na))
    >>>             stat[action] = (Na[action] - 1) / Na[action] * stat[action] + reward / Na[
    >>>                 action
    >>>             ]
    >>>             return stat
    >>>
    >>>         def index(stat, Na, t):
    >>>             return stat + np.sqrt( np.log(t**2) / (2*Na))
    >>>
    >>>         RecursiveIndexAgent.__init__(self, env, stat_function, index, **kwargs)
    """

    name = "RecursiveIndexAgent"

    def __init__(self, env, stat_function=None, index_function=None, **kwargs):
        BanditWithSimplePolicy.__init__(self, env, **kwargs)
        self.n_arms = self.env.action_space.n
        if stat_function is None:

            def stat_function_(stat, Na, action, reward):
                if stat is None:
                    stat = np.zeros(len(Na))
                stat[action] = (Na[action] - 1) / Na[action] * stat[
                    action
                ] + reward / Na[action]
                return stat

            self.stat_function = stat_function_
        else:
            self.stat_function = stat_function
        if index_function is None:

            def index(stat, Na, t):
                return stat + np.sqrt(np.log(t**2) / (2 * Na))

            self.index_function = index
        else:
            self.index_function = index_function
        self.total_time = 0

    def fit(self, budget=None, **kwargs):
        horizon = budget
        indices = np.inf * np.ones(self.n_arms)
        stats = None
        Na = np.zeros(self.n_arms)
        total_reward = 0
        for ep in range(horizon):
            if self.total_time < self.n_arms:
                action = self.total_time
            else:
                indices = self.get_recursive_indices(stats, Na, ep)
                action = np.argmax(indices)
            self.total_time += 1
            Na[action] += 1
            _, reward, _, _ = self.env.step(action)
            stats = self.stat_function(stats, Na, action, reward)
            total_reward += reward

        self.optimal_action = np.argmax(indices)

        info = {"episode_reward": total_reward}
        return info

    def get_recursive_indices(self, stats, Na, ep):
        indices = np.zeros(self.n_arms)
        for a in self.arms:
            indices[a] = self.index_function(stats[a], Na[a], ep)
        return indices
