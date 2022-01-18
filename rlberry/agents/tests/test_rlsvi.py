import pytest
from rlberry.agents.rlsvi import RLSVIAgent
from rlberry.envs.finite import GridWorld


@pytest.mark.parametrize("gamma, stage_dependent",
                         [
                             (1.0, True),
                             (1.0, False),
                             (0.9, True),
                             (0.9, False),
                         ])
def test_rlsvi(gamma, stage_dependent):
    env = GridWorld(walls=(), nrows=5, ncols=5)
    agent = RLSVIAgent(env,
                       horizon=11,
                       stage_dependent=stage_dependent,
                       gamma=gamma)
    agent.fit(budget=50)
    agent.policy(env.observation_space.sample())