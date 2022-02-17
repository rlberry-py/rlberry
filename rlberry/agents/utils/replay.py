"""
New module aiming to replace memories.py
"""

import numpy as np


class ReplayBuffer:
    """Simple replay buffer that allows sampling data with shape (batch_size, time_size, ...).

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
    >>>
    >>> # Sample a batch of 32 sub-trajectories of length 100
    >>> batch = buffer.sample(batch_size=32, chunk_size=100)
    >>> for tag in batch:
    >>>     print(tag, batch[tag].shape)
    """

    def __init__(self, max_replay_size, rng, max_episode_steps=None):
        self._rng = rng
        self._max_replay_size = max_replay_size
        self._tags = []
        self._data = dict()
        self._dtypes = dict()
        self._max_episode_steps = max_episode_steps

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
            Data type of the entry (e.g. `np.float32`)
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
        assert list(data.keys()) == self.tags
        for tag in self.tags:
            self._data[tag].append(data[tag])
        if len(self) > self._max_replay_size:
            for tag in self.tags:
                self._data[tag].pop(0)

    def _sample_one_chunk(self, chunk_size):
        current_size = len(self)
        if chunk_size != -1:
            start_index = self._rng.choice(current_size - chunk_size)
            end_index = start_index + chunk_size
        else:
            assert self._max_episode_steps is not None
            assert self._max_episode_steps is not np.inf

            if "dones" not in self.tags:
                raise RuntimeError(
                    "To sample from the replay with chunk_size=-1, "
                    "terminal-state flags (done) must be stored in the "
                    "replay under the tag `dones`"
                )

            chunk_size = self._max_episode_steps
            all_dones = 1.0 * np.array(self._data["dones"])
            trajectory_ends_indices = np.nonzero(all_dones == 1.0)[0] + 1
            # take only indices >= chunk size
            trajectory_ends_indices = trajectory_ends_indices[
                trajectory_ends_indices >= chunk_size
            ]
            end_index = self._rng.choice(trajectory_ends_indices)
            start_index = end_index - chunk_size
            assert start_index >= 0

        chunk = dict()
        for tag in self.tags:
            chunk[tag] = np.array(self._data[tag][start_index:end_index])
        return chunk

    def __len__(self):
        return len(self._data[self.tags[0]])

    def sample(self, batch_size, chunk_size):
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
            Length of each sub-trajectory. If -1, it is set to self.max_episode_steps,
            and the last observation in the sampled sub-trajectories corresponds to
            a terminal state.
        """
        if len(self) <= chunk_size:
            return None
        batch = dict()

        trajectories = dict()
        for tag in self.tags:
            trajectories[tag] = []
        for _ in range(batch_size):
            chunk = self._sample_one_chunk(chunk_size)
            for tag in self.tags:
                trajectories[tag].append(chunk[tag])
        for tag in self.tags:
            batch[tag] = np.array(trajectories[tag], dtype=self._dtypes[tag])
        return batch
