# based on https://github.com/openai/gym/blob/master/tests/utils/test_env_checker.py

import numpy as np
import pytest
from rlberry.envs import Chain
from gym.utils.env_checker import check_env


class TestEnv(Chain):
    def step(self, action):
        observation = np.array([1.0, 1.5, 0.5])
        reward = 1
        done = True
        return observation, reward, done


def test_check_env_dict_action():
    # Environment.step() only returns 3 values: obs, reward, done. Not info!
    test_env = TestEnv()

    with pytest.raises(AssertionError) as errorinfo:
        check_env(env=test_env, warn=True)
        assert (
            str(errorinfo.value)
            == "The `step()` method must return four values: obs, reward, done, info"
        )
