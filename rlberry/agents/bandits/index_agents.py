import numpy as np
from rlberry.agents import AgentWithSimplePolicy
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IndexAgent(AgentWithSimplePolicy):
    """
    Agent for bandit environment using Index-based policy.

    Parameters
    -----------
    env : rlberry bandit environment
        See :class:`~rlberry.envs.bandits.Bandit`.

    index_function : callable or None, default = None
        Compute the index for an arm using the past rewards on this arm and
        the current time t. If None, use UCB bound for bernoulli.


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

    def __init__(self, env, index_function=lambda rew, t: np.mean(rew), **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.n_arms = self.env.action_space.n
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

        indexes = np.inf * np.ones(self.n_arms)
        for ep in range(horizon):
            if self.total_time < self.n_arms:
                action = self.total_time
            else:
                indexes = self.get_indexes(rewards, actions, ep)
                action = np.argmax(indexes)
            self.total_time += 1
            _, reward, _, _ = self.env.step(action)
            rewards[ep] = reward
            actions[ep] = action

        self.optimal_action = np.argmax(indexes)
        info = {"episode_reward": np.sum(rewards)}
        return info

    def get_indexes(self, rewards, actions, ep):
        """
        Return the indexes of each arms.

        Parameters
        ----------

        rewards : array, shape (n_iterations,)
            list of rewards until now

        actions : array, shape (n_iterations,)
            list of actions until now

        ep: int
            current iteration/epoch

        """
        indexes = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            indexes[a] = self.index_function(rewards[actions == a], ep)
        return indexes

    def policy(self, observation):
        return self.optimal_action

    def save(self, filename):
        """
        Save agent object.

        Parameters
        ----------
        filename: Path or str
            File in which to save the Agent.

        Returns
        -------
        If save() is successful, a Path object corresponding to the filename is returned.
        Otherwise, None is returned.
        Important: the returned filename might differ from the input filename: For instance,
        the method can append the correct suffix to the name before saving.

        """

        dico = {
            "_writer": self.writer,
            "seeder": self.seeder,
            "_execution_metadata": self._execution_metadata,
            "_unique_id": self._unique_id,
            "_output_dir": self._output_dir,
            "optimal_action": self.optimal_action,
        }

        # save
        filename = Path(filename).with_suffix(".pickle")
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open("wb") as ff:
            pickle.dump(dico, ff)

        return filename

    @classmethod
    def load(cls, filename, **kwargs):
        """Load agent object.

        If overridden, save() method must also be overriden.

        Parameters
        ----------
        **kwargs: dict
            Arguments to required by the __init__ method of the Agent subclass.
        """
        filename = Path(filename).with_suffix(".pickle")

        obj = cls(**kwargs)
        with filename.open("rb") as ff:
            tmp_dict = pickle.load(ff)

        obj.__dict__.update(tmp_dict)

        return obj


class RecursiveIndexAgent(AgentWithSimplePolicy):
    """
    Agent for bandit environment using a recursive Index-based policy.

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

    def __init__(self, env, stat_function, index_function, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
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
        rewards = np.zeros(horizon)
        actions = np.ones(horizon) * np.nan
        indexes = np.inf * np.ones(self.n_arms)
        stats = None
        Na = np.zeros(self.n_arms)
        for ep in range(horizon):
            if self.total_time < self.n_arms:
                action = self.total_time
            else:
                indexes = self.get_recursive_indexes(stats, Na, ep)
                action = np.argmax(indexes)
            self.total_time += 1
            Na[action] += 1
            _, reward, _, _ = self.env.step(action)
            stats = self.stat_function(stats, Na, action, reward)
            rewards[ep] = reward
            actions[ep] = action

        self.optimal_action = np.argmax(indexes)

        info = {"episode_reward": np.sum(rewards)}
        return info

    def get_recursive_indexes(self, stats, Na, ep):
        indexes = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            indexes[a] = self.index_function(stats[a], Na[a], ep)
        return indexes

    def policy(self, observation):
        return self.optimal_action

    def save(self, filename):
        """
        Save agent object.

        Parameters
        ----------
        filename: Path or str
            File in which to save the Agent.

        Returns
        -------
        If save() is successful, a Path object corresponding to the filename is returned.
        Otherwise, None is returned.
        Important: the returned filename might differ from the input filename: For instance,
        the method can append the correct suffix to the name before saving.

        """
        # remove writer if not pickleable

        dico = {
            "_writer": self.writer,
            "seeder": self.seeder,
            "_execution_metadata": self._execution_metadata,
            "_unique_id": self._unique_id,
            "_output_dir": self._output_dir,
            "optimal_action": self.optimal_action,
        }

        # save
        filename = Path(filename).with_suffix(".pickle")
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open("wb") as ff:
            pickle.dump(dico, ff)

        return filename

    @classmethod
    def load(cls, filename, **kwargs):
        """Load agent object.

        If overridden, save() method must also be overriden.

        Parameters
        ----------
        **kwargs: dict
            Arguments to required by the __init__ method of the Agent subclass.
        """
        filename = Path(filename).with_suffix(".pickle")

        obj = cls(**kwargs)
        with filename.open("rb") as ff:
            tmp_dict = pickle.load(ff)

        obj.__dict__.update(tmp_dict)

        return obj
