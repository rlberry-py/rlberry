import numpy as np
import pytest
from rlberry.agents.features import FeatureMap
from rlberry.agents.linear.lsvi_ucb import LSVIUCBAgent
from rlberry.envs.finite import GridWorld


class OneHotFeatureMap(FeatureMap):
    def __init__(self, S, A):
        self.S = S
        self.A = A
        self.shape = (S*A,)

    def map(self, observation, action):
        feat = np.zeros((self.S, self.A))
        feat[observation, action] = 1.0
        return feat.flatten()


class RandomFeatMap(FeatureMap):
    def __init__(self, S, A):
        self.feat_mat = np.random.randn(S, A, 10)
        self.shape = (10,)

    def map(self, observation, action):
        feat = self.feat_mat[observation, action, :]
        return feat.copy()


@pytest.mark.parametrize("FeatMapClass", [OneHotFeatureMap, RandomFeatMap])
def test_lsvi_ucb_matrix_inversion(FeatMapClass):
    env = GridWorld(nrows=3, ncols=3, walls=())

    def feature_map_fn():
        return FeatMapClass(env.observation_space.n, env.action_space.n)

    agent = LSVIUCBAgent(env, n_episodes=10,
                         feature_map_fn=feature_map_fn,
                         horizon=10)
    agent.fit()
    assert np.allclose(np.linalg.inv(agent.lambda_mat), agent.lambda_mat_inv)
    assert agent.episode == 10
    agent.policy(env.observation_space.sample())
