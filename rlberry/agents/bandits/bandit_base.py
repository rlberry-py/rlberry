import numpy as np
from rlberry.agents import AgentWithSimplePolicy
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BanditTracker:
    """
    Container class for rewards and various statistics (means...) collected
    during the run of a bandit algorithm.

    Parameters
    ----------

    n_arms: int.
        Number of arms.

    params: dict
        Other parameters to condition what to store and compute.
    """

    name = "BanditTracker"

    def __init__(self, agent, params={}):
        self.n_arms = agent.n_arms
        self.seeder = agent.seeder
        # Store rewards for each arm or not
        self.store_rewards = params.get("store_rewards", False)
        # Add importance weighted rewards or not
        self.do_iwr = params.get("do_iwr", False)
        self.reset()

    def reset(self):
        self.S_hats = np.zeros(self.n_arms)
        self.mu_hats = np.zeros(self.n_arms)
        self.n_pulls = np.zeros(self.n_arms, dtype="int")
        self.t = 0
        if self.store_rewards:
            self.rewards = [[] for _ in range(self.n_arms)]
        if self.do_iwr:
            self.iw_S_hats = np.zeros(self.n_arms)

    def update(self, arm, reward, params={}):
        self.t += 1
        self.n_pulls[arm] += 1
        self.S_hats[arm] += reward
        self.mu_hats[arm] = self.S_hats[arm] / self.n_pulls[arm]
        if self.store_rewards:
            self.rewards[arm].append(reward)
        if self.do_iwr:
            p = params.get("p", 1.0)
            self.iw_S_hats[arm] += 1 - (1 - reward) / p


class BanditWithSimplePolicy(AgentWithSimplePolicy):
    """
    Base class for bandits algorithms.

    The fit function must result in self.optimal_action being set for the save
    and load functions to work.

    Parameters
    -----------
    env: rlberry bandit environment
        See :class:`~rlberry.envs.bandits.Bandit`.

    tracker_params: dict
        Parameters for the tracker object, typically to decide what to store.

    """

    name = ""

    def __init__(self, env, tracker_params={}, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.n_arms = self.env.action_space.n
        self.arms = np.arange(self.n_arms)
        self.tracker = BanditTracker(self, tracker_params)

    @property
    def total_time(self):
        return self.tracker.t

    def fit(self, budget=None, **kwargs):
        """
        Example fit function. Should be overwritten by your own implementation.

        Parameters
        ----------
        budget: int
            Total number of iterations, also called horizon.
        """
        horizon = budget
        rewards = np.zeros(horizon)

        for ep in range(horizon):
            # choose the optimal action
            # for demo purpose, we will always choose action 0
            action = 0
            _, reward, _, _ = self.env.step(action)
            self.tracker.update(action, reward)
            rewards[ep] = reward

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
