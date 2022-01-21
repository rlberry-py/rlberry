import numpy as np
from rlberry.agents import AgentWithSimplePolicy
import dill
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

    index_function : callable, default = lambda rew, t : np.mean(rew)
        Compute the index for an arm using the past rewards on this arm and
        the current time t.

    recursive_index_function : tuple of callable or None
        Compute recursively a sufficient statistic with the first member of the
        tuple, the second member is the size of this stat and the last compute the
        index using the sufficient statistics.

    record_action : bool, default=False
        If True, record in the writer the number of time an action is played at
        the end of each simulation.

    phase : bool, default = None
        If True, compute "phased bandit" where the index is computed only every
        phase**j. If None, the Bandit is not phased.
    """

    name = "IndexAgent"

    def __init__(
        self,
        env,
        index_function=lambda rew, t: np.mean(rew),
        recursive_index_function=None,
        record_action=False,
        phase=None,
        **kwargs
    ):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.n_arms = self.env.action_space.n
        self.index_function = index_function
        self.recursive_index_function = recursive_index_function
        self.record_action = record_action
        self.phase = phase
        self.total_time = 0

    def fit(self, budget=None, **kwargs):
        n_episodes = budget
        rewards = np.zeros(n_episodes)
        actions = np.ones(n_episodes) * np.nan

        indexes = np.inf * np.ones(self.n_arms)
        for ep in range(n_episodes):
            self.total_time += 1
            if self.total_time < self.n_arms:
                action = self.global_time
            else:
                indexes = self.get_indexes(rewards, actions, ep)
                action = np.argmax(indexes)
                next_state, reward, done, _ = self.env.step(action)
                rewards[ep] = reward
                actions[ep] = action

        self.optimal_action = np.argmax(indexes)
        Na = [np.sum(actions == a) for a in range(self.n_arms)]
        if self.record_action:
            for a in range(self.n_arms):
                self.writer.add_scalar("episode_Na" + str(a), Na[a], 1)
        info = {"episode_reward": np.sum(rewards)}
        return info

    def get_indexes(self, rewards, actions, ep):
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
        # remove writer if not pickleable
        if not dill.pickles(self.writer):
            self.set_writer(None)

        dict = {
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
        try:
            with filename.open("wb") as ff:
                pickle.dump(dict, ff)
        except Exception:
            try:
                with filename.open("wb") as ff:
                    dill.dump(self.dict, ff)
            except Exception as ex:
                logger.warning("Agent instance cannot be pickled: " + str(ex))
                return None

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
        try:
            with filename.open("rb") as ff:
                tmp_dict = pickle.load(ff)
        except Exception:
            with filename.open("rb") as ff:
                tmp_dict = dill.load(ff)

        obj.__dict__.update(tmp_dict)

        return obj
