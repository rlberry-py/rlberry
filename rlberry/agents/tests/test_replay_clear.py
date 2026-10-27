import numpy as np
import pytest

from rlberry.agents.utils.replay import ReplayBuffer


def make_buffer(capacity, prioritized):
    buffer = ReplayBuffer(
        capacity, np.random.default_rng(7), enable_prioritized=prioritized
    )
    buffer.setup_entry("reward", np.float32)
    return buffer


@pytest.mark.parametrize("capacity", [8, 10])
@pytest.mark.parametrize("prioritized", [False, True])
def test_clear_resets_replay_bookkeeping(capacity, prioritized):
    buffer = make_buffer(capacity, prioritized)
    for i in range(capacity + 3):
        buffer.append({"reward": float(i)})
    buffer.end_episode()
    if prioritized:
        buffer.update_priorities(np.array([[0, 1]]), np.array([[25.0, 100.0]]))
    buffer.clear()
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.tags == ["reward"]
    assert buffer.dtypes == {"reward": np.float32}
    assert buffer._position == 0
    assert buffer._episodes == []
    assert buffer._current_episode == 0
    if prioritized:
        assert buffer._max_priority == 1.0
        assert buffer._it_sum.sum() == 0
        assert buffer._it_min.min() == float("inf")


@pytest.mark.parametrize("mode", ["uniform", "prioritized"])
def test_cleared_buffer_can_be_reused_like_fresh_buffer(mode):
    buffer = make_buffer(8, True)
    reference = make_buffer(8, True)
    for i in range(10):
        buffer.append({"reward": -float(i)})
    buffer.update_priorities(np.array([[0, 1, 7]]), np.array([[10.0, 100.0, 1000.0]]))
    buffer.end_episode()
    buffer.clear()
    for i in range(5):
        buffer.append({"reward": float(i)})
        reference.append({"reward": float(i)})
    actual = buffer.sample(16, 2, sampling_mode=mode)
    expected = reference.sample(16, 2, sampling_mode=mode)
    np.testing.assert_array_equal(actual.data["reward"], expected.data["reward"])
    for key in ["indices", "weights"]:
        np.testing.assert_array_equal(actual.info[key], expected.info[key])
