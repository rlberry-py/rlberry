from time import time

from rlberry.agents.torch import DQNAgent
from rlberry.envs import gym_make
from rlberry.manager import AgentManager


def test_multiprocessing():
    """Check if multiprocessing is efficient."""
    env_ctor = gym_make
    env_kwargs = dict(id="Acrobot-v1")

    dqn_init_kwargs = dict(
        gamma=0.99,
        batch_size=32,
        chunk_size=8,
        lambda_=0.5,
        target_update_parameter=0.005,
        learning_rate=1e-3,
        epsilon_init=1.0,
        epsilon_final=0.1,
        epsilon_decay_interval=20_000,
        train_interval=10,
        gradient_steps=-1,
        max_replay_size=200_000,
        learning_starts=5_000,
    )

    agent1 = AgentManager(
        DQNAgent,
        (env_ctor, env_kwargs),
        init_kwargs=dqn_init_kwargs,
        fit_budget=10000,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=4,
        parallelization="process",
    )

    agent2 = AgentManager(
        DQNAgent,
        (env_ctor, env_kwargs),
        init_kwargs=dqn_init_kwargs,
        fit_budget=10000,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=2,
        parallelization="process",
    )

    start = time()
    agent1.fit()
    end = time()
    agent1_time = end - start
    print(agent1_time)

    start = time()
    agent2.fit()
    end = time()
    agent2_time = end - start
    print(2*agent2_time)

    assert (
        agent1_time < 2*agent2_time
    ), f"The execution time of agent 1 ({agent1_time}), should be lower than the execution time of the agent 2 ({agent2_time})"
