import numpy as np
from rlberry.agents.utils import replay
from rlberry.envs.finite import GridWorld
from gym.wrappers import TimeLimit


def test_replay():
    batch_size = 64
    chunk_size = 128

    env = GridWorld(terminal_states=None)
    env = TimeLimit(env, max_episode_steps=200)
    env.reseed(123)

    rng = np.random.default_rng(456)
    buffer = replay.ReplayBuffer(
        500, rng, max_episode_steps=env._max_episode_steps, enable_prioritized=True
    )
    buffer.setup_entry("observations", np.float32)
    buffer.setup_entry("actions", np.uint32)
    buffer.setup_entry("rewards", np.float32)
    buffer.setup_entry("dones", np.bool)

    # Fill the replay buffer
    total_time = 0
    while True:
        if total_time > 2 * buffer._max_replay_size:
            break
        done = False
        obs = env.reset()
        while not done:
            total_time += 1
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
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

    # Sample batches
    for _ in range(10):
        batch = buffer.sample(
            batch_size=batch_size, chunk_size=chunk_size, sampling_mode="uniform"
        )
        batch_prioritized = buffer.sample(
            batch_size=batch_size, chunk_size=chunk_size, sampling_mode="prioritized"
        )

        for test_batch in [batch, batch_prioritized]:
            for tag in buffer.tags:
                assert test_batch.data[tag].shape[:2] == (batch_size, chunk_size)
                assert test_batch.data[tag].dtype == buffer.dtypes[tag]
                assert np.array_equal(
                    np.array(buffer.data[tag], dtype=buffer.dtypes[tag])[
                        test_batch.info["indices"]
                    ],
                    test_batch.data[tag],
                )

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
