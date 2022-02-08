import numpy as np


class SimpleReplayBuffer:
    """Simple replay buffer that allows sampling data with shape (batch_size, time_size, ...)."""

    def __init__(self, max_replay_size, rng, max_episode_steps=None):
        self._rng = rng
        self._max_replay_size = max_replay_size
        self._tags = []
        self._data = dict()
        self._dtypes = dict()
        self._max_episode_steps = max_episode_steps

    @property
    def tags(self):
        return self._tags

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
        self._data[tag] = []
        self._dtypes[tag] = dtype

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

            chunk_size = self._max_episode_steps
            all_discounts = np.array(self._discounts)
            trajectory_ends_indices = np.nonzero(all_discounts == 0)[0] + 1
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


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import pickle
#     from numpy.random import default_rng

#     replay = SimpleReplayBuffer(max_replay_size=50000, rng=default_rng())
#     replay.setup_entry("state", np.float32)

#     for _ in range(2):
#         initial_buffer_size = len(replay)
#         sampled_from_replay = []
#         for ii in range(10000):
#             state = (ii + initial_buffer_size) * np.ones((2,), dtype=np.float32)
#             state[1] += 0.1
#             replay.append({"state": state})
#             batch = replay.sample(1, 1)
#             if batch:
#                 sampled_from_replay.append(batch["state"][0][0][0])

#         plt.figure()
#         plt.plot(sampled_from_replay)

#         print("-------------------------------------------------")
#         print("save and load replay")
#         pickle.dump(replay, open("temp/saved_replay.pickle", "wb"))
#         del replay
#         replay = pickle.load(open("temp/saved_replay.pickle", "rb"))

#     plt.show()
