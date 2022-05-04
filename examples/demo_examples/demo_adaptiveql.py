""" 
 ===================== 
 Demo: demo_adaptiveql 
 =====================
"""
import numpy as np
from rlberry.agents import AdaptiveQLAgent
from rlberry.agents import RSUCBVIAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.manager import (
    MultipleManagers,
    AgentManager,
    plot_writer_data,
    evaluate_agents,
)
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = (get_benchmark_env, dict(level=2))

    N_EP = 1000
    HORIZON = 30

    params = {}
    params["adaql"] = {"horizon": HORIZON, "gamma": 1.0, "bonus_scale_factor": 1.0}

    params["rsucbvi"] = {
        "horizon": HORIZON,
        "gamma": 1.0,
        "bonus_scale_factor": 1.0,
        "min_dist": 0.05,
        "max_repr": 800,
    }

    eval_kwargs = dict(eval_horizon=HORIZON, n_simulations=20)

    multimanagers = MultipleManagers(parallelization="thread")
    multimanagers.append(
        AgentManager(
            AdaptiveQLAgent,
            env,
            fit_budget=N_EP,
            init_kwargs=params["adaql"],
            eval_kwargs=eval_kwargs,
            n_fit=4,
            output_dir="dev/examples/",
        )
    )
    multimanagers.append(
        AgentManager(
            RSUCBVIAgent,
            env,
            fit_budget=N_EP,
            init_kwargs=params["rsucbvi"],
            n_fit=2,
            output_dir="dev/examples/",
        )
    )

    multimanagers.run(save=False)

    evaluate_agents(multimanagers.managers)

    plot_writer_data(
        multimanagers.managers,
        tag="episode_rewards",
        preprocess_func=np.cumsum,
        title="Cumulative Rewards",
    )

    for stats in multimanagers.managers:
        agent = stats.get_agent_instances()[0]
        try:
            agent.Qtree.plot(0, 25)
        except AttributeError:
            pass
    plt.show()

    for stats in multimanagers.managers:
        print(f"Agent = {stats.agent_name}, Eval = {stats.eval_agents()}")
