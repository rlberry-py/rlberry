import pytest
from rlberry.agents.psrl import PSRLAgent
from rlberry.envs.finite import GridWorld


@pytest.mark.parametrize("gamma, stage_dependent, bernoullized_reward",
                         [
                             (1.0, True, True),
                             (1.0, True, False),
                             (1.0, False, True),
                             (1.0, False, False),
                             (0.9, True, True),
                             (0.9, True, False),
                             (0.9, False, True),
                             (0.9, False, False),
                         ])
def test_ucbvi(gamma, stage_dependent, bernoullized_reward):
    env = GridWorld(walls=(), nrows=5, ncols=5)
    agent = PSRLAgent(
        env,
        horizon=11,
        bernoullized_reward=bernoullized_reward,
        stage_dependent=stage_dependent,
        gamma=gamma)
    agent.fit(budget=50)
    agent.policy(env.observation_space.sample())
