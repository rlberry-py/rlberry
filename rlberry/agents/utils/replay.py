"""
New module aiming to replace memories.py
"""

import numpy as np
from typing import NamedTuple
from rlberry.agents.utils import replay_utils


class Batch(NamedTuple):
    data: dict
    info: dict


class ReplayBuffer:
    """Replay buffer that allows sampling data with shape (batch_size, time_size, ...).

    Notes
    -----
    For prioritized experience replay, code was adapted from
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    Parameters
    ----------
    max_replay_size: int
        Maximum number of transitions that can be stored
    rng: Generator
        Numpy random number generator.
        See https://numpy.org/doc/stable/reference/random/generator.html
    max_episode_steps: int, optional
        Maximum length of an episode
    enable_prioritized: bool, default = False
        If True, enable sampling with prioritized experience replay,
        by setting sampling_mode="prioritized" in the :meth:`sample` method.
    alpha: float, default = 0.5
        How much prioritization is used, if prioritized=True,
        (0 - no prioritization, 1 - full prioritization).
    beta: float, default = 0.5
        To what degree to use importance weights, if prioritized=True,
        (0 - no corrections, 1 - full correction).

    Examples
    --------
    >>> import numpy as np
    >>> from rlberry.agents.utils import replay
    >>> from rlberry.envs import gym_make
    >>>
    >>> rng = np.random.default_rng()
    >>> buffer = replay.ReplayBuffer(100_000, rng)
    >>> buffer.setup_entry("observations", np.float32)
    >>> buffer.setup_entry("actions", np.uint32)
    >>> buffer.setup_entry("rewards", np.uint32)
    >>>
    >>> # Store data in the replay
    >>> env = gym_make("CartPole-v0")
    >>> for _ in range(500):
    >>>     done = False
    >>>     obs = env.reset()
    >>>     while not done:
    >>>         action = env.action_space.sample()
    >>>         next_obs, reward, done, info = env.step(action)
    >>>         buffer.append(
    >>>             {
    >>>                 "observations": obs,
    >>>                 "actions": action,
    >>>                 "rewards": reward
    >>>             }
    >>>         )
    >>>         obs = next_obs
    >>>         if done:
    >>>             buffer.end_episode()
    >>> # Sample a batch of 32 sub-trajectories of length 100
    >>> batch = buffer.sample(batch_size=32, chunk_size=100)
    >>> for tag in buffer.tags:
    >>>     print(tag, batch.data[tag].shape)
    """

    def __init__(
        self,
        max_replay_size,
        rng,
        max_episode_steps=None,
        enable_prioritized=False,
        alpha=0.5,
        beta=0.5,
    ):
        self._rng = rng
        self._max_replay_size = max_replay_size
        self._tags = []
        self._data = dict()
        self._dtypes = dict()
        self._max_episode_steps = max_episode_steps
        self._episodes = []
        self._current_episode = 0
        self._position = 0

        self._enable_prioritized = enable_prioritized
        self._alpha = alpha
        self._beta = beta
        if self._enable_prioritized:
            it_capacity = 1
            while it_capacity < self._max_replay_size:
                it_capacity *= 2

            self._it_sum = replay_utils.SumSegmentTree(it_capacity)
            self._it_min = replay_utils.MinSegmentTree(it_capacity)
            self._max_priority = 1.0

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
    def max_episode_steps(self):
        """Maximum length of an episode."""
        return self._max_episode_steps

    def clear(self):
        """Clear data in replay."""
        for tag in self._data:
            self._data[tag] = []

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
        if tag in self._data:
            raise ValueError(f"Entry {tag} already added to replay buffer.")
        self._tags.append(tag)
        self._dtypes[tag] = dtype
        self._data[tag] = []

    def append(self, data):
        """Store data.

        Parameters
        ----------
        data : dict
            Dictionary containing scalar values, whose keys must be in self.tags.
        """
        #
        # Append data
        #
        assert list(data.keys()) == self.tags
        if len(self) < self._max_replay_size:
            self._episodes.append(self._current_episode)
            for tag in self.tags:
                self._data[tag].append(data[tag])
        else:
            position = self._position
            self._episodes[position] = self._current_episode
            for tag in self.tags:
                self._data[tag][position] = data[tag]

        #
        # Priorities
        #
        if self._enable_prioritized:
            self._it_sum[self._position] = self._max_priority**self._alpha
            self._it_min[self._position] = self._max_priority**self._alpha

        # Faster than append and pop
        self._position = (self._position + 1) % self._max_replay_size

    def end_episode(self):
        """Call this method to indicate the end of an episode."""
        self._current_episode += 1

    def _sample_proportional_and_compute_weights(self, max_val, batch_size):
        """Used for prioritized replay."""
        # sample indices
        indices = []
        p_total = self._it_sum.sum(0, max_val - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = self._rng.uniform() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            indices.append(idx)
        indices = np.array(indices)

        # importance weights
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-self._beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-self._beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        return indices, weights

    def _sample_batch_indices(self, batch_size, chunk_size, sampling_mode):
        current_size = len(self)
        if sampling_mode == "uniform":
            start_indices = self._rng.choice(current_size - chunk_size, size=batch_size)
            weights = np.ones_like(np.indices, dtype=np.float32)
        elif sampling_mode == "prioritized":
            start_indices, weights = self._sample_proportional_and_compute_weights(
                current_size - chunk_size, batch_size=batch_size
            )
            assert np.all(start_indices <= (current_size - chunk_size))
        else:
            raise ValueError(f"Invalid sampling_mode: '{sampling_mode}'")
        end_indices = start_indices + chunk_size

        # Handle invalid sub-trajectories: those containing a transition
        # corresponding to self._position. That is, we need to handle
        # the fact that we're sampling sub-trajectories from a circular buffer
        # and that we cannot cross the 'boundary' defined by self._position.
        #
        # What we do here is to move the start_index to the most recent
        # index (with respect to self._position) that allows sampling
        # a sub-trajectory of length chunk_size.
        # Note that this can result in a negative start_index,
        # which is handled in the _get_chunk method.
        bad_indices = np.logical_and(
            start_indices < self._position, end_indices >= (self._position + 1)
        )
        start_boundary = self._position - chunk_size
        start_indices[bad_indices] = start_boundary
        end_indices[bad_indices] = start_boundary + chunk_size
        is_biased = bad_indices
        return start_indices, end_indices, weights, is_biased

    def _get_chunk(self, start_index, end_index):
        """Note: start_index can be negative!"""
        chunk = dict()
        for tag in self.tags:
            if start_index >= 0:
                data = self._data[tag][start_index:end_index]
            else:
                data = self._data[tag][start_index:] + self._data[tag][:end_index]
            chunk[tag] = np.array(data)
        indices = np.arange(start_index, end_index)
        return chunk, indices

    def __len__(self):
        return len(self._data[self.tags[0]])

    def sample(self, batch_size, chunk_size, sampling_mode="uniform"):
        """Sample a batch.

        Data have shape (B, T, ...), where
        B = batch_size
        T = chunk_size
        and represents a batch of sub-trajectories.

        Parameters
        ----------
        batch_size: int
            Number of sub-trajectories to sample.
        chunk_size: int
            Length of each sub-trajectory.
        sampling_mode: {"uniform", "prioritized"}, default = "uniform"
            "uniform": sample batch uniformly at random;
            "prioritized": use prioritized experience replay (requires
            enable_prioritized=True in the constructor).
        """
        assert sampling_mode in ["uniform", "prioritized"]
        if sampling_mode == "prioritized" and not self._enable_prioritized:
            raise RuntimeError(
                "To sample from the replay with the 'prioritized' mode, "
                "you need to set enable_prioritized=True in the constructor."
            )

        if len(self) <= chunk_size:
            return None
        # sample start/end indices for sub-trajectories
        start_indices, end_indices, weights, is_biased = self._sample_batch_indices(
            batch_size, chunk_size, sampling_mode
        )

        batch_data = dict()
        batch_info = dict()

        trajectories = dict()
        all_indices = []
        for tag in self.tags:
            trajectories[tag] = []

        for ii in range(batch_size):
            chunk, indices = self._get_chunk(start_indices[ii], end_indices[ii])
            for tag in self.tags:
                trajectories[tag].append(chunk[tag])
            all_indices.append(indices)

        for tag in self.tags:
            batch_data[tag] = np.array(trajectories[tag], dtype=self._dtypes[tag])

        batch_info["indices"] = np.array(all_indices, dtype=np.int)
        batch_info["weights"] = weights
        batch_info["is_biased"] = is_biased

        batch = Batch(data=batch_data, info=batch_info)
        return batch

    def update_priorities(self, indices, new_priorities):
        """Update priorities in the replay buffer.

        Parameters
        ----------
        indices : array of shape (batch, time)
            Numpy array containing indices of transitions to be updated.
            From a sampled batch, you can set it to batch.info["indices"].
        new_priorities : array of shape (batch, time)
            Numpy array containing the new priorities. Must have the same
            shape as the indices array.
        """
        assert self._enable_prioritized

        if indices.shape != new_priorities.shape:
            raise RuntimeError(
                "Shape of 'indices' does not match shape of `new_priorities`"
            )

        batch_dim, time_dim = indices.shape
        for bb in range(batch_dim):
            for tt in range(time_dim):
                idx = indices[bb, tt]
                priority = new_priorities[bb, tt]
                assert 0 <= idx < len(self)
                self._it_sum[idx] = priority**self._alpha
                self._it_min[idx] = priority**self._alpha
                self._max_priority = max(self._max_priority, priority)
