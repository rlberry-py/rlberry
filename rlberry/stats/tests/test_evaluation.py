import numpy as np
from rlberry.stats.evaluation import mc_policy_evaluation
from rlberry.envs.finite import GridWorld
from rlberry.agents.dynprog.value_iteration import ValueIterationAgent


def test_mc_policy_eval(gamma=0.5, horizon=20, stationary_policy=True):
    env = GridWorld(nrows=3, ncols=3,
                    start_coord=(0, 0),
                    success_probability=1.0,
                    walls=(), default_reward=0.0,
                    reward_at={(2, 2): 1.0})
    agent = ValueIterationAgent(env, gamma=gamma, horizon=horizon)
    agent.fit()

    episode_rewards = mc_policy_evaluation(agent, env, n_sim=5, gamma=gamma, stationary_policy=stationary_policy)
    assert episode_rewards.mean() == 1.0 * np.power(gamma, 4)


test_mc_policy_eval()
