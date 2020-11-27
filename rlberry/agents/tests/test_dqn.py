import gym
from rlberry.wrappers import Wrapper
from rlberry.agents.dqn.pytorch import DQNAgent


def test_dqn_agent():
    env = Wrapper(gym.make("CartPole-v0"))
    agent = DQNAgent(env, {"n_episodes": 10})
    agent.fit()
    agent.policy(env.observation_space.sample())
