from rlberry.agents.jax.dqn.dqn import DQNAgent
from rlberry.envs import gym_make
from rlberry.stats import AgentStats


def test_jax_dqn():
    env = (gym_make, dict(id='CartPole-v0'))
    params = dict(
        chunk_size=4,
        batch_size=128,
        target_update_interval=5,
    )

    stats = AgentStats(
        DQNAgent,
        env,
        fit_budget=20,
        eval_env=env,
        init_kwargs=params,
        n_fit=1,
        parallelization='thread',
    )
    stats.fit()
    stats.clear_output_dir()
