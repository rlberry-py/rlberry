import numpy as np
import pytest
from rlberry.stats.evaluation import mc_policy_evaluation
from rlberry.envs.finite import GridWorld
from rlberry.agents.dynprog.value_iteration import ValueIterationAgent


@pytest.mark.parametrize(
    "gamma, horizon, stationary_policy",
    [
        (1.0, 20, True),
        (1.0, 20, False),
        (0.5, 20, True),
        (0.5, 20, False),
    ]
    )
def test_mc_policy_eval(gamma, horizon, stationary_policy):
    env = GridWorld(nrows=3, ncols=3,
                    start_coord=(0, 0),
                    success_probability=1.0,
                    walls=(), default_reward=0.0,
                    reward_at={(2, 2): 1.0})
    agent = ValueIterationAgent(env, gamma=gamma, horizon=horizon)
    agent.fit()

    episode_rewards = mc_policy_evaluation(agent, env, n_sim=5, gamma=gamma, stationary_policy=stationary_policy)
    assert episode_rewards.mean() == 1.0 * np.power(gamma, 4)


@pytest.mark.parametrize(
    "gamma, horizon, stationary_policy",
    [
        (1.0, 20, True),
        (1.0, 20, False),
        (0.5, 20, True),
        (0.5, 20, False),
    ]
    )
def test_mc_policy_eval_2(gamma, horizon, stationary_policy):
    env = GridWorld(nrows=3, ncols=3,
                    start_coord=(0, 0),
                    success_probability=1.0,
                    walls=(), default_reward=0.0,
                    reward_at={(2, 2): 1.0})
    agent1 = ValueIterationAgent(env, gamma=gamma, horizon=horizon)
    agent2 = ValueIterationAgent(env, gamma=gamma, horizon=horizon)

    agent1.fit()
    agent2.fit()

    agents = [agent1, agent2]

    episode_rewards = mc_policy_evaluation(agents, env, n_sim=5, gamma=gamma, stationary_policy=stationary_policy)
    assert episode_rewards.mean() == 1.0 * np.power(gamma, 4)
