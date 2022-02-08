import numpy as np
import collections


ReplayData = collections.namedtuple(
    "ReplayData",
    ["observations", "actions", "rewards", "discounts", "next_observations"],
)


class ReplayBuffer:
    def __init__(self, max_replay_size, rng, max_episode_steps=None):
        self._rng = rng
        self._max_replay_size = max_replay_size
        self._observations = []
        self._actions = []
        self._rewards = []
        self._discounts = []
        self._next_observations = []

        self._max_episode_steps = max_episode_steps

    @property
    def data(self):
        observations = self._observations
        actions = self._actions
        rewards = self._rewards
        discounts = self._discounts
        next_observations = self._next_observations
        return ReplayData(observations, actions, rewards, discounts, next_observations)

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
        observations = np.array(self._observations[start_index:end_index])
        actions = np.array(self._actions[start_index:end_index])
        rewards = np.array(self._rewards[start_index:end_index])
        discounts = np.array(self._discounts[start_index:end_index])
        next_observations = np.array(self._next_observations[start_index:end_index])

        return observations, actions, rewards, discounts, next_observations

    def __len__(self):
        return len(self._rewards)

    def append(self, obs, action, reward, discount, next_obs):
        self._observations.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        self._discounts.append(discount)
        self._next_observations.append(next_obs)
        if len(self) > self._max_replay_size:
            self._observations.pop(0)
            self._actions.pop(0)
            self._rewards.pop(0)
            self._discounts.pop(0)
            self._next_observations.pop(0)

    def sample(self, batch_size, chunk_size):
        """
        Data have shape (B, T, ...), where
        B = batch_size
        T = chunk_size
        and represents a batch of sub-trajectories.
        """
        if len(self) <= chunk_size:
            return None
        batch = dict()
        obs_trajectories = []
        action_trajectories = []
        reward_trajectories = []
        discount_trajectories = []
        next_obs_trajectories = []
        for _ in range(batch_size):
            (
                observations,
                actions,
                rewards,
                discounts,
                next_obs,
            ) = self._sample_one_chunk(chunk_size)
            obs_trajectories.append(observations)
            action_trajectories.append(actions)
            reward_trajectories.append(rewards)
            discount_trajectories.append(discounts)
            next_obs_trajectories.append(next_obs)
        batch["observations"] = np.array(obs_trajectories)
        batch["actions"] = np.array(action_trajectories)
        batch["rewards"] = np.array(reward_trajectories, dtype=np.float32)
        batch["discounts"] = np.array(discount_trajectories, dtype=np.float32)
        batch["dones"] = batch["discounts"] == 0.0
        batch["next_observations"] = np.array(next_obs_trajectories)
        return batch


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pickle
    from numpy.random import default_rng

    replay = ReplayBuffer(max_replay_size=50000, rng=default_rng())
    for _ in range(2):
        initial_buffer_size = len(replay)
        sampled_from_replay = []
        for ii in range(10000):
            state = (ii + initial_buffer_size) * np.ones((2,), dtype=np.float32)
            state[1] += 0.1
            replay.append(state, 1, 2, 3, 4)
            batch = replay.sample(1, 1)
            if batch:
                sampled_from_replay.append(batch["observations"][0][0][0])

        plt.figure()
        plt.plot(sampled_from_replay)

        print("-------------------------------------------------")
        print("save and load replay")
        pickle.dump(replay, open("temp/saved_replay.pickle", "wb"))
        del replay
        replay = pickle.load(open("temp/saved_replay.pickle", "rb"))

    plt.show()
