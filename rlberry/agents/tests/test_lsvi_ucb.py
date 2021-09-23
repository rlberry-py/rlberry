import numpy as np
import pytest
from rlberry.agents.features import FeatureMap
from rlberry.agents.linear.lsvi_ucb import LSVIUCBAgent
from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.envs.finite import GridWorld


class OneHotFeatureMap(FeatureMap):
    def __init__(self, S, A):
        self.S = S
        self.A = A
        self.shape = (S * A,)

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
    env.reseed(123)

    def feature_map_fn(_env):
        return FeatMapClass(_env.observation_space.n, _env.action_space.n)

    reg_factor = 0.1
    agent = LSVIUCBAgent(env,
                         feature_map_fn=feature_map_fn,
                         horizon=10,
                         reg_factor=reg_factor)
    agent.reseed(123)
    agent.fit(budget=50)
    assert np.allclose(np.linalg.inv(agent.lambda_mat), agent.lambda_mat_inv)
    assert agent.episode == 50
    agent.policy(env.observation_space.sample())

    # Check counts
    if FeatMapClass != OneHotFeatureMap:
        return

    S = env.observation_space.n
    A = env.action_space.n
    N_sa = np.zeros((S, A))
    for state, action in zip(agent.state_hist, agent.action_hist):
        N_sa[state, action] += 1.0

    assert np.allclose(agent.lambda_mat_inv.diagonal(),
                       1.0 / (N_sa.flatten() + reg_factor))

    for ss in range(S):
        for aa in range(A):
            feat = agent.feature_map.map(ss, aa)
            assert np.allclose(feat @ (agent.lambda_mat_inv.T @ feat),
                               1.0 / (N_sa[ss, aa] + reg_factor))


def test_lsvi_without_bonus():
    def lsvi_debug_gather_data(agent):
        """
        Function to gather data sampling uniformly
        states and actions
        """
        N = agent.n_episodes * agent.horizon
        count = 0
        while count < N:
            state = agent.env.observation_space.sample()
            action = agent.env.action_space.sample()
            next_state, reward, done, info = agent.env.sample(state, action)
            #
            #
            feat = agent.feature_map.map(state, action)
            outer_prod = np.outer(feat, feat)
            inv = agent.lambda_mat_inv

            #
            agent.lambda_mat += np.outer(feat, feat)
            # update inverse
            agent.lambda_mat_inv -= \
                (inv @ outer_prod @ inv) / (1 + feat @ inv.T @ feat)

            # update history
            agent.reward_hist[count] = reward
            agent.state_hist.append(state)
            agent.action_hist.append(action)
            agent.nstate_hist.append(next_state)

            #
            tt = agent.total_time_steps
            agent.feat_hist[tt, :] = agent.feature_map.map(state, action)
            for aa in range(agent.env.action_space.n):
                agent.feat_ns_all_actions[tt, aa, :] = \
                    agent.feature_map.map(next_state, aa)

            # increments
            agent.total_time_steps += 1
            count += 1

    env = GridWorld(nrows=2, ncols=2, walls=(), success_probability=0.95)
    env.reseed(123)

    def feature_map_fn(_env):
        return OneHotFeatureMap(_env.observation_space.n, _env.action_space.n)

    agent = LSVIUCBAgent(env,
                         feature_map_fn=feature_map_fn,
                         horizon=20,
                         gamma=0.99,
                         reg_factor=1e-5)
    agent.reseed(123)
    agent.n_episodes = 100
    agent.reset()

    lsvi_debug_gather_data(agent)
    # estimated Q
    S = env.observation_space.n
    Q_est = agent._run_lsvi(bonus_factor=0.0)[0, :].reshape((S, -1))

    # near optimal Q
    agent_opt = ValueIterationAgent(env, gamma=0.99, horizon=20)
    agent_opt.fit()
    Q = agent_opt.Q[0, :, :]

    print(Q)
    print("---")
    print(Q_est)

    print("-------")
    print(np.abs(Q - Q_est))
    # Check error
    assert Q_est == pytest.approx(Q, rel=0.01)


def test_lsvi_random_exploration():
    env = GridWorld(nrows=2, ncols=2, walls=(), success_probability=0.95)
    env.reseed(123)

    def feature_map_fn(_env):
        return OneHotFeatureMap(_env.observation_space.n, _env.action_space.n)

    agent = LSVIUCBAgent(env,
                         feature_map_fn=feature_map_fn,
                         horizon=20,
                         gamma=0.99,
                         reg_factor=1e-5,
                         bonus_scale_factor=0.0)
    agent.reseed(123)
    agent.fit(budget=250)

    # estimated Q
    S = env.observation_space.n
    Q_est = agent._run_lsvi(bonus_factor=0.0)[0, :].reshape((S, -1))

    # near optimal Q
    agent_opt = ValueIterationAgent(env, gamma=0.99, horizon=20)
    agent_opt.fit()
    Q = agent_opt.Q[0, :, :]

    print(Q)
    print("---")
    print(Q_est)

    print("-------")
    print(np.abs(Q - Q_est))
    # Check error
    assert np.abs(Q - Q_est).mean() < 0.1


def test_lsvi_optimism():
    env = GridWorld(nrows=2, ncols=2, walls=())

    def feature_map_fn(_env):
        return OneHotFeatureMap(_env.observation_space.n, _env.action_space.n)

    agent = LSVIUCBAgent(env, gamma=0.99,
                         feature_map_fn=feature_map_fn,
                         horizon=3,
                         bonus_scale_factor=3,
                         reg_factor=0.000001)
    agent.fit(budget=250)

    # near optimal Q
    agent_opt = ValueIterationAgent(env, gamma=0.99, horizon=3)
    agent_opt.fit()
    Q = agent_opt.Q[0, :, :]

    # optimistic Q
    S = env.observation_space.n
    A = env.action_space.n
    Q_optimistic = np.zeros((S, A))
    for ss in range(S):
        Q_optimistic[ss, :] = agent._compute_q_vec(
            agent.w_vec[0, :],
            ss,
            agent.bonus_scale_factor)

    print(Q)
    print(Q_optimistic)
    assert (Q_optimistic - Q).min() >= -1e-5
