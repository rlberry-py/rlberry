import copy
import logging

import gym
import numpy as np

from rlberry.envs.utils import process_env
from rlberry.utils.jit_setup import numba_jit


logger = logging.getLogger(__name__)


def process_ppo_env(env, seeder, num_envs=1, asynchronous=False):
    """
    Process environment for PPO. It is the only agent that supports vectorized
    environments.

    Parameters
    ----------
    env : gym.Env
        Environment to be processed.
    seeder : rlberry.Seeder
        Seeder object.
    num_envs : int
        Number of environments to be used.
    asynchronous : bool
        If True, the environments are run asynchronously.

    Returns
    -------
    vec_env : gym.vector.VectorEnv
        Vectorized environment.
    """
    vec_env_cls = (
        gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    )
    return vec_env_cls(
        [lambda: process_env(env, seeder, copy_env=True) for _ in range(num_envs)]
    )


@numba_jit
def lambda_returns(r_t, terminal_tp1, v_tp1, gamma, lambda_):
    """
    Compute lambda returns.

    Parameters
    ----------
    r_t: array
        Array of shape (time_dim, batch_dim) containing the rewards.
    terminal_tp1: array
        Array of shape (time_dim, batch_dim) containing the discounts (0.0 if terminal state).
    v_tp1: array
        Array of shape (time_dim, batch_dim) containing the values at timestep t+1
    lambda_ : float in [0, 1]
        Lambda-returns parameter.
    """
    T = v_tp1.shape[0]
    returns = np.zeros_like(r_t)
    aux = v_tp1[-1].astype(np.float32)
    for tt in range(T):
        i = T - tt - 1
        returns[i] = r_t[i] + gamma * (1 - terminal_tp1[i]) * (
            (1 - lambda_) * v_tp1[i] + lambda_ * aux
        )
        aux = returns[i]
    return returns


class RolloutBuffer:
    """
    Rollout buffer that allows sampling data with shape (batch_size,
    num_trajectories, ...).
    Parameters
    ----------
    rng: numpy.random.Generator
        Numpy random number generator.
        See https://numpy.org/doc/stable/reference/random/generator.html
    max_episode_steps: int, optional
        Maximum length of an episode
    """

    def __init__(self, rng, num_rollout_steps):
        self._rng = rng
        self._num_rollout_steps = num_rollout_steps
        self._curr_step = 0
        self._tags = []
        self._data = dict()
        self._dtypes = dict()

    @property
    def data(self):
        """Dict containing all stored data."""
        return self._data

    @property
    def tags(self):
        """Tags identifying the entries in the replay buffer."""
        return self._tags

    @property
    def dtypes(self):
        """Dict containing the data types for each tag."""
        return self._dtypes

    @property
    def num_rollout_steps(self):
        """Number of steps to take in each environment per policy rollout."""
        return self._num_rollout_steps

    @property
    def num_envs(self):
        return self._num_envs

    def __len__(self):
        return self._curr_step

    def full(self):
        """Returns True if the buffer is full."""
        return len(self) == self.num_rollout_steps

    def clear(self):
        """Clear data in replay."""
        self._curr_step = 0
        for tag in self._data:
            self._data[tag] = None

    def setup_entry(self, tag, dtype):
        """Configure replay buffer to store data.
        Parameters
        ----------
        tag : str
            Tag that identifies the entry (e.g "observation", "reward")
        dtype : obj
            Data type of the entry (e.g. `np.float32`). Type is not
            checked in :meth:`append`, but it is used to construct the numpy
            arrays returned by the :meth:`sample`method.
        """
        assert len(self) == 0, "Cannot setup entry on non-empty buffer."
        if tag in self._data:
            raise ValueError(f"Entry {tag} already added to replay buffer.")
        self._tags.append(tag)
        self._dtypes[tag] = dtype
        self._data[tag] = None

    def append(self, data):
        """
        Stores data from an environment step in the buffer.

        Parameters
        ----------
        data : dict
            Dictionary containing scalar values, whose keys must be in self.tags.
        """
        assert set(data.keys()) == set(self.tags), "Data keys must be in self.tags"
        assert len(self) < self.num_rollout_steps, "Buffer is full."
        for tag in self.tags:
            #
            if self._data[tag] is None:
                if isinstance(data[tag], np.ndarray):
                    # if data[tag].dtype != self._dtypes[tag]:
                    #     logger.warning(
                    #         f"Data type for tag {tag} is {data[tag].dtype}, "
                    #         f"but it was configured as {self._dtypes[tag]}.")
                    shape = data[tag].shape
                    self._data[tag] = np.zeros(
                        (self.num_rollout_steps, *shape), dtype=self._dtypes[tag]
                    )
                elif isinstance(data[tag], float) or isinstance(data[tag], int):
                    self._data[tag] = np.zeros(
                        self.num_rollout_steps, dtype=self._dtypes[tag]
                    )
                else:
                    self._data[tag] = [None] * self.num_rollout_steps
            self._data[tag][self._curr_step] = data[tag]
        self._curr_step += 1

    def get(self):
        """
        Returns the collected data. If the appended data for a given tag is a
        numpy array, the returned data will be a numpy array of shape:

        (T, *S), where T is the number of rollout steps, and S is the shape of
        the data that was appended.

        Otherwise, the returned data will be a list of length T.

        Returns
        -------
        Returns a dict with the collected data.
        """
        return copy.deepcopy(self._data)
