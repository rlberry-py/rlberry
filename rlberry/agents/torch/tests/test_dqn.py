from rlberry.envs import gym_make
from rlberry.agents.torch.dqn import DQNAgent
from rlberry.exploration_tools.online_discretization_counter import OnlineDiscretizationCounter
from rlberry.exploration_tools.torch.rnd import RandomNetworkDistillation
from rlberry.seeding import Seeder
import numpy as np


def test_dqn_agent():
    env = gym_make("MountainCar-v0")

    def uncertainty_estimator_fn(observation_space, action_space):
        counter = OnlineDiscretizationCounter(
            observation_space,
            action_space,
            min_dist=0.25)
        return counter

    agent = DQNAgent(env,
                     use_bonus=True,
                     uncertainty_estimator_kwargs=dict(
                         uncertainty_estimator_fn=uncertainty_estimator_fn,
                         bonus_scale_factor=1.0
                     ))
    agent.fit(budget=5)
    agent.policy(env.observation_space.sample())

    # test seeding of exploration policy
    agent2 = DQNAgent(env,
                      use_bonus=True,
                      uncertainty_estimator_kwargs=dict(
                          uncertainty_estimator_fn=uncertainty_estimator_fn,
                          bonus_scale_factor=1.0
                      ))
    agent.reseed(Seeder(123))
    agent2.reseed(Seeder(123))

    n1 = agent.exploration_policy.np_random.integers(2 ** 32, size=2)
    n2 = agent2.exploration_policy.np_random.integers(2 ** 32, size=2)
    assert np.array_equal(n1, n2)


def test_dqn_agent_rnd():
    env = gym_make("CartPole-v0")

    def uncertainty_estimator_fn(observation_space, action_space):
        counter = RandomNetworkDistillation(observation_space, action_space)
        return counter

    agent = DQNAgent(env,
                     use_bonus=True,
                     uncertainty_estimator_kwargs=dict(
                         uncertainty_estimator_fn=uncertainty_estimator_fn,
                         bonus_scale_factor=1.0
                     ))
    agent.fit(budget=5)
    agent.policy(env.observation_space.sample())
