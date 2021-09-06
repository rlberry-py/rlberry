
import rlberry.agents.jax.nets.common as nets
from rlberry.agents.jax.dqn.dqn import DQNAgent
from rlberry.envs import gym_make
from rlberry.stats import AgentStats, MultipleStats, plot_writer_data


if __name__ == '__main__':

    # global params
    fit_budget = 10000
    n_fit = 2

    # env and algorithm params
    env = (gym_make, dict(id='CartPole-v0'))
    params = dict(
        chunk_size=8,
        batch_size=64,
        target_update_interval=500,
        eval_interval=200,
        gamma=0.975,
        lambda_=0.5,
        learning_rate=0.0015,
        net_constructor=nets.MLPQNetwork,
        net_kwargs=dict(
            num_actions=env[0](**env[1]).action_space.n,
            hidden_sizes=(64, 64)
        )
    )

    params_alternative = params.copy()
    params_alternative.update(
        dict(
            net_kwargs=dict(
                num_actions=env[0](**env[1]).action_space.n,
                hidden_sizes=(16, 16)
            )
        )
    )

    stats = AgentStats(
        DQNAgent,
        env,
        fit_budget=fit_budget,
        eval_env=env,
        init_kwargs=params,
        n_fit=n_fit,
        parallelization='process',
        agent_name='dqn',
    )

    stats_alternative = AgentStats(
        DQNAgent,
        env,
        fit_budget=fit_budget,
        eval_env=env,
        init_kwargs=params_alternative,
        n_fit=n_fit,
        parallelization='process',
        agent_name='dqn_smaller_net'
    )

    # fit everything in parallel
    mstats = MultipleStats()
    mstats.append(stats)
    mstats.append(stats_alternative)
    mstats.run()

    plot_writer_data(mstats.allstats, tag='episode_rewards', show=False)
    plot_writer_data(mstats.allstats, tag='dw_time_elapsed', show=False)
    plot_writer_data(mstats.allstats, tag='eval_rewards', show=False)
    plot_writer_data(mstats.allstats, tag='q_loss')

    stats.save()
    stats.clear_output_dir()
