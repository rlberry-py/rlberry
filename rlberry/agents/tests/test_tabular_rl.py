import pytest

from rlberry.agents import QLAgent, SARSAAgent
from rlberry.envs import GridWorld


@pytest.mark.parametrize(
    "exploration_type, exploration_rate",
    [("epsilon", 0.5), ("boltzmann", 0.5), (None, None)],
)
def test_ql(exploration_type, exploration_rate):
    env = GridWorld(walls=(), nrows=5, ncols=5)
    agent = QLAgent(
        env, exploration_type=exploration_type, exploration_rate=exploration_rate
    )
    agent.fit(budget=50)
    agent.policy(env.observation_space.sample())
    agent.reset()
    assert not agent.Q.any()


@pytest.mark.parametrize(
    "exploration_type, exploration_rate",
    [("epsilon", 0.5), ("boltzmann", 0.5), (None, None)],
)
def test_sarsa(exploration_type, exploration_rate):
    env = GridWorld(walls=(), nrows=5, ncols=5)
    agent = SARSAAgent(
        env, exploration_type=exploration_type, exploration_rate=exploration_rate
    )
    agent.fit(budget=50)
    agent.policy(env.observation_space.sample())
    agent.reset()
    assert not agent.Q.any()
