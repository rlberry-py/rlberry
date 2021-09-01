
from rlberry.agents.jax.dqn.dqn import DQNAgent
from rlberry.envs import gym_make
from rlberry.stats import AgentStats, plot_writer_data


if __name__ == '__main__':
    env = (gym_make, dict(id='CartPole-v0'))
    params = dict(
        chunk_size=4,
        batch_size=128,
        target_update_interval=1000,
    )

    stats = AgentStats(
        DQNAgent,
        env,
        fit_budget=1000,
        eval_env=env,
        init_kwargs=params,
        n_fit=4,
        parallelization='process',
    )

    stats.fit()       # fit with fit_budget
    stats.fit(10000)    # fit with another budget

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
    plot_writer_data([stats], tag='q_loss')

    stats.save()

    stats.optimize_hyperparams(
        timeout=500,
        n_fit=1,
        n_optuna_workers=1,
        sampler_method='random',
        optuna_parallelization='process')
    stats.clear_output_dir()
