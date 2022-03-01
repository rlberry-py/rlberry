import numpy as np
from rlberry.agents import AgentWithSimplePolicy
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BanditWithSimplePolicy(AgentWithSimplePolicy):
    """
    Base class for bandits algorithms.

    The fit function must result in self.optimal_action being set for the save
    and load functions to work.

    Parameters
    -----------
    env : rlberry bandit environment
        See :class:`~rlberry.envs.bandits.Bandit`.

    """

    name = ""

    def __init__(self, env, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.n_arms = self.env.action_space.n
        self.arms = np.arange(self.n_arms)
        self.total_time = 0

    def fit(self, budget=None, **kwargs):
        """
        Example fit function. Should be overwritten by your own implementation.
        """
        horizon = budget
        rewards = np.zeros(horizon)
        actions = np.ones(horizon) * np.nan

        for ep in range(horizon):
            # choose the optimal action
            # for demo purpose, we will always choose action 0
            action = 0
            self.total_time += 1
            _, reward, _, _ = self.env.step(action)
            rewards[ep] = reward
            actions[ep] = action

        self.optimal_action = 0
        info = {"episode_reward": np.sum(rewards)}
        return info

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
