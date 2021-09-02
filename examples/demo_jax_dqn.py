
from rlberry.agents.jax.dqn.dqn import DQNAgent
from rlberry.envs import gym_make
from rlberry.stats import AgentStats, plot_writer_data


if __name__ == '__main__':
    env = (gym_make, dict(id='CartPole-v0'))
    params = dict(
        chunk_size=8,
        batch_size=128,
        target_update_interval=2000,
        eval_interval=200,
        lambda_=0.1,
    )

    stats = AgentStats(
        DQNAgent,
        env,
        fit_budget=20000,
        eval_env=env,
        init_kwargs=params,
        n_fit=2,
        parallelization='process',
    )
    stats.fit()

    agent = stats.agent_handlers[0]
    env_instance = stats.agent_handlers[0].env
    obs = env_instance.reset()
    for _ in range(10):
        action = agent.policy(obs)
        obs, _, done, _ = env_instance.step(action)
        env_instance.render()
        if done:
            break
    env_instance.close()

    plot_writer_data([stats], tag='episode_rewards', show=False)
    plot_writer_data([stats], tag='dw_time_elapsed', show=False)
    plot_writer_data([stats], tag='eval_rewards', show=False)
    plot_writer_data([stats], tag='q_loss')

    stats.save()
    stats.clear_output_dir()
