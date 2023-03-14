import pytest
import numpy as np
from rlberry.agents.utils import replay
from rlberry.envs.finite import GridWorld
from gymnasium.wrappers import TimeLimit

def _get_filled_replay(max_replay_size):
    """runs env for ~ 2 * max_replay_size timesteps."""
    env = GridWorld(terminal_states=None)
    env = TimeLimit(env, max_episode_steps=200)                  #use gymnasium wrapper
    env.reseed(123)

    rng = np.random.default_rng(456)
    buffer = replay.ReplayBuffer(
        max_replay_size,
        rng,
        max_episode_steps=env._max_episode_steps,  #inside the 'TimeLimit' wrapper
        enable_prioritized=True,
    )
    buffer.setup_entry("observations", np.float32)
    buffer.setup_entry("actions", np.uint32)
    buffer.setup_entry("rewards", np.float32)
    buffer.setup_entry("dones", bool)

    # Fill the replay buffer
    total_time = 0
    while True:
        if total_time > 2 * buffer._max_replay_size:
            break
        done = False
        observation,info = env.reset()
        while not done:
            total_time += 1
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.append(
                {
                    "observations": observation,
                    "actions": action,
                    "rewards": reward,
                    "dones": done,
                }
            )
            observation = next_observation
            if done:
                buffer.end_episode()
    return buffer, env


def test_replay_size():
    # get replay buffer
    buffer, _ = _get_filled_replay(max_replay_size=500)
    assert len(buffer) == 500


@pytest.mark.parametrize("sampling_mode", ["uniform", "prioritized"])
def test_replay_sampling(sampling_mode):
    batch_size = 128
    chunk_size = 256

    # get replay buffer
    buffer, _ = _get_filled_replay(max_replay_size=500)

    # Sample batches, check shape and dtype
    for _ in range(10):
        batch = buffer.sample(
            batch_size=batch_size, chunk_size=chunk_size, sampling_mode=sampling_mode
        )
        for tag in buffer.tags:
            assert batch.data[tag].shape[:2] == (batch_size, chunk_size)
            assert batch.data[tag].dtype == buffer.dtypes[tag]
            assert np.array_equal(
                np.array(buffer.data[tag], dtype=buffer.dtypes[tag])[
                    batch.info["indices"]
                ],
                batch.data[tag],
            )


def test_replay_priority_update():
    # get replay buffer
    buffer, _ = _get_filled_replay(max_replay_size=500)
    rng = buffer._rng
    # Test priority update
    # Note: the test only works if all indices in indices_to_update are unique (otherwise the
    # same priority index can be updated more than once, which will break the test)
    indices_to_update = rng.choice(len(buffer), size=(32, 10), replace=False)
    n_indices = indices_to_update.shape[0] * indices_to_update.shape[1]
    new_priorities = np.arange(n_indices).reshape(indices_to_update.shape)
    buffer.update_priorities(indices_to_update.copy(), new_priorities.copy())

    for bb in range(indices_to_update.shape[0]):
        for cc in range(indices_to_update.shape[1]):
            idx = indices_to_update[bb, cc]
            val1 = buffer._it_sum[idx]
            val2 = buffer._it_min[idx]
            assert val1 == val2 == new_priorities[bb, cc] ** buffer._alpha


@pytest.mark.parametrize("sampling_mode", ["uniform", "prioritized"])
def test_replay_samples_valid_indices(sampling_mode):
    batch_size = 16
    chunk_size = 256

    # get replay buffer
    buffer, env = _get_filled_replay(max_replay_size=500)

    # add more data, sample batches and check that sampled sub-trajetories
    # are not "crossing" the current position (buffer._position)
    total_time = 0
    while True:
        if total_time > 1000:
            break
        done = False
        obs, info = env.reset()
        while not done:
            total_time += 1
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.append(
                {
                    "observations": obs,
                    "actions": action,
                    "rewards": reward,
                    "dones": done,
                }
            )
            obs = next_obs
            if done:
                buffer.end_episode()

            # sample and check
            start_indices, end_indices, weights = buffer._sample_batch_indices(
                batch_size, chunk_size, sampling_mode=sampling_mode
            )
            assert np.all(weights >= 0), "weights must be nonnegative"

            # we need end_indices > start_indices and the difference
            # to be equal to chunk_size
            assert np.all((end_indices - start_indices) == chunk_size)

            positive_mask = start_indices >= 0
            negative_mask = ~positive_mask

            # Case 1: start indices are >= 0
            assert np.all(
                ~np.logical_and(
                    buffer._position > start_indices[positive_mask],
                    buffer._position < end_indices[positive_mask],
                )
            ), "buffer._position cannot be in the middle of start and end indices"
            # Case 2: start indices are < 0
            # -> self._position cannot be between start_indices+len(buffer) and len(buffer)-1
            # -> self._position cannot be between 0 and end_indices-1
            assert np.all(
                np.logical_and(
                    (start_indices[negative_mask] + len(buffer)) >= buffer._position,
                    end_indices[negative_mask] <= buffer._position,
                )
            ), "buffer._position cannot be in the middle of start and end indices"
