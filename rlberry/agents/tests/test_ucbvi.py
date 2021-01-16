import pytest
from rlberry.agents.ucbvi import UCBVIAgent
from rlberry.envs.finite import GridWorld


@pytest.mark.parametrize("gamma, stage_dependent, real_time_dp",
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
def test_ucbvi(gamma, stage_dependent, real_time_dp):
    env = GridWorld(walls=(), nrows=5, ncols=5)
    agent = UCBVIAgent(env,
                       n_episodes=50,
                       horizon=11,
                       stage_dependent=stage_dependent,
                       gamma=gamma,
                       real_time_dp=real_time_dp,
                       bonus_scale_factor=0.1)
    agent.fit()
    agent.policy(env.observation_space.sample())
