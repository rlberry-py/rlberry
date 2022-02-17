import numpy as np
from rlberry.agents.utils import replay
from rlberry.envs import gym_make


def test_replay():
    env = gym_make("CartPole-v0")

    rng = np.random.default_rng()
    buffer = replay.ReplayBuffer(100_000, rng, max_episode_steps=env._max_episode_steps)
    buffer.setup_entry("observations", np.float32)
    buffer.setup_entry("actions", np.uint32)
    buffer.setup_entry("rewards", np.float32)
    buffer.setup_entry("dones", np.bool)

    # Store data in the replay
    for _ in range(100):
        done = False
        obs = env.reset()
        while not done:
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

    # Sample a batch of 32 sub-trajectories of length 10
    batch = buffer.sample(batch_size=32, chunk_size=10)
    batch_full_traj = buffer.sample(batch_size=32, chunk_size=-1)
    for tag in buffer.tags:
        assert batch[tag].shape[:2] == (32, 10)
        assert batch[tag].dtype == buffer.dtypes[tag]
        assert np.array_equal(
            np.array(buffer.data[tag], dtype=buffer.dtypes[tag])[batch["indices"]],
            batch[tag]
        )

        assert batch_full_traj[tag].shape[:2] == (32, env._max_episode_steps)


    assert np.all(batch_full_traj["dones"][:, -1])


test_replay()