from rlberry.envs import gym_make
from rlberry.agents.dqn.pytorch import DQNAgent


def test_dqn_agent():
    env = gym_make("CartPole-v0")
    params = {"n_episodes": 10}
    agent = DQNAgent(env, **params)
    agent.fit()
    agent.policy(env.observation_space.sample())
