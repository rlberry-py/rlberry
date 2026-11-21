import gymnasium as gym
import numpy as np
import pytest

from rlberry.wrappers import DiscretizeStateWrapper


class KeywordResetEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
        self.action_space = gym.spaces.Discrete(2)
        self.last_seed = None
        self.last_options = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_seed = seed
        self.last_options = options
        observation = self.np_random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        return observation, {"seed": seed}


@pytest.mark.parametrize("n_bins", [1, 5])
@pytest.mark.parametrize("seed", [None, 42])
@pytest.mark.parametrize("options", [None, {"start": 0.25}])
def test_discretize_reset_forwards_keywords(n_bins, seed, options):
    raw_env = KeywordResetEnv()
    env = DiscretizeStateWrapper(raw_env, n_bins)
    observation, info = env.reset(seed=seed, options=options)
    assert env.observation_space.contains(observation)
    assert raw_env.last_seed == seed
    assert raw_env.last_options is options
    assert info == {"seed": seed}
    if seed is not None:
        expected, _ = KeywordResetEnv().reset(seed=seed, options=options)
        assert observation == env.get_discrete_state(expected)
        assert env.reset(seed=seed, options=options)[0] == observation
    env.close()


@pytest.mark.parametrize("n_bins", [1, 5])
@pytest.mark.parametrize("options", [None, {"low": -0.55, "high": -0.45}])
def test_discretize_gymnasium_mountain_car(n_bins, options):
    raw_env = gym.make("MountainCar-v0")
    reference = gym.make("MountainCar-v0")
    env = DiscretizeStateWrapper(raw_env, n_bins)
    observation, info = env.reset(seed=42, options=options)
    expected, expected_info = reference.reset(seed=42, options=options)
    assert env.observation_space.contains(observation)
    assert observation == env.get_discrete_state(expected)
    assert info == expected_info
    for action in [0, 2, 1]:
        observation, *result = env.step(action)
        expected, *expected_result = reference.step(action)
        assert observation == env.get_discrete_state(expected)
        assert result == expected_result
    env.close()
    reference.close()
