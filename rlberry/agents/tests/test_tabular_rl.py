from rlberry.agents import QLAgent, SARSAAgent
from rlberry.envs import GridWorld


def test_ql():
    env = GridWorld()
    agent = QLAgent(env, epsilon=0.3)
    agent.fit(budget=2)


def test_sarsa():
    env = GridWorld()
    agent = SARSAAgent(env, epsilon=0.3)
    agent.fit(budget=2)
