import gym

from rlberry.agents.dqn.pytorch import DQNAgent


def test_dqn_agent():
    env = gym.make("CartPole-v0")
    agent = DQNAgent(env, {"n_episodes": 10})
    agent.fit()
    agent.policy(env.observation_space.sample())
