import pytest

from rlberry.envs.classic_control import MountainCar
from rlberry.wrappers.discretize_state import DiscretizeStateWrapper


@pytest.mark.parametrize("n_bins", list(range(1, 10)))
def test_discretizer(n_bins):
    env = DiscretizeStateWrapper(MountainCar(), n_bins)
    assert env.observation_space.n == n_bins * n_bins

    for ep in range(2):
        state = env.reset()
        for ii in range(50):
            assert env.observation_space.contains(state)
            action = env.action_space.sample()
            next_s, reward, done, info = env.step(action)
            state = next_s

    for ii in range(100):
        state = env.observation_space.sample()
        action = env.action_space.sample()
        next_s, reward, done, info = env.sample(state, action)
        assert env.observation_space.contains(next_s)

    assert env.unwrapped.id == "MountainCar"
