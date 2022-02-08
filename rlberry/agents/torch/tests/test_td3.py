import pytest
from rlberry.envs import gym_make
from rlberry.agents.torch.td3 import nets as td3nets
from rlberry.agents.torch.td3.td3 import TD3Agent


def q_net_constructor(env):
    return td3nets.TD3MLPCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_sizes=(256, 256),
    )


def pi_net_constructor(env):
    return td3nets.TD3MLPActor(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_sizes=(256, 256),
    )


@pytest.mark.parametrize("env_id", ["CartPole-v0", "Pendulum-v1"])
def test_td3_agent(env_id):
    env = (gym_make, dict(id=env_id))
    params = dict(
        q_net_constructor=q_net_constructor,
        pi_net_constructor=pi_net_constructor,
        learning_starts=1,
        train_interval=-1,
        gamma=0.99,
    )
    agent = TD3Agent(env, **params)
    agent.reseed(123)
    agent.fit(20)
    agent.policy(agent.env.observation_space.sample())
