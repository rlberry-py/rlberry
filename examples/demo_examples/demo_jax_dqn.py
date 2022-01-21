""" 
 ===================== 
 Demo: demo_jax_dqn 
 =====================
"""
import rlberry.agents.jax.nets.common as nets
from rlberry.agents.jax.dqn.dqn import DQNAgent
from rlberry.envs import gym_make
from rlberry.manager import AgentManager, MultipleManagers, plot_writer_data

if __name__ == "__main__":
    # global params
    fit_budget = 10000
    n_fit = 2

    # env and algorithm params
    env = (gym_make, dict(id="CartPole-v0"))
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
            num_actions=env[0](**env[1]).action_space.n, hidden_sizes=(64, 64)
        ),
    )

    params_alternative = params.copy()
    params_alternative.update(
        dict(
            net_kwargs=dict(
                num_actions=env[0](**env[1]).action_space.n, hidden_sizes=(16, 16)
            )
        )
    )

    stats = AgentManager(
        DQNAgent,
        env,
        fit_budget=fit_budget,
        eval_env=env,
        init_kwargs=params,
        n_fit=n_fit,
        parallelization="process",
        agent_name="dqn",
    )

    stats_alternative = AgentManager(
        DQNAgent,
        env,
        fit_budget=fit_budget,
        eval_env=env,
        init_kwargs=params_alternative,
        n_fit=n_fit,
        parallelization="process",
        agent_name="dqn_smaller_net",
    )

    # fit everything in parallel
    multimanagers = MultipleManagers()
    multimanagers.append(stats)
    multimanagers.append(stats_alternative)
    multimanagers.run()

    plot_writer_data(multimanagers.managers, tag="episode_rewards", show=False)
    plot_writer_data(multimanagers.managers, tag="dw_time_elapsed", show=False)
    plot_writer_data(multimanagers.managers, tag="eval_rewards", show=False)
    plot_writer_data(multimanagers.managers, tag="q_loss")

    stats.save()
    stats.clear_output_dir()
