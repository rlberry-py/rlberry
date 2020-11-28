import gym
from rlberry.wrappers import Wrapper
from rlberry.agents.dqn.pytorch import DQNAgent


def test_dqn_agent():
    env = Wrapper(gym.make("CartPole-v0"))
    params = {"n_episodes": 10}
    agent = DQNAgent(env, **params)
    agent.fit()
    agent.policy(env.observation_space.sample())
