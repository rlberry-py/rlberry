"""
 =====================
 Demo: run_remote_manager
 =====================
"""
from rlberry.envs.gym_make import gym_make
from rlberry.network.client import BerryClient
from rlberry.network.interface import ResourceRequest

from rlberry.agents.torch import REINFORCEAgent

from rlberry.manager.agent_manager import AgentManager
from rlberry.manager.multiple_managers import MultipleManagers
from rlberry.manager.remote_agent_manager import RemoteAgentManager
from rlberry.manager.evaluation import evaluate_agents, plot_writer_data


if __name__ == "__main__":
    port = int(input("Select server port: "))
    client = BerryClient(port=port)

    FIT_BUDGET = 500

    local_manager = AgentManager(
        agent_class=REINFORCEAgent,
        train_env=(gym_make, dict(id="CartPole-v1")),
        fit_budget=FIT_BUDGET,
        init_kwargs=dict(gamma=0.99),
        eval_kwargs=dict(eval_horizon=200, n_simulations=20),
        n_fit=2,
        seed=10,
        agent_name="REINFORCE(local)",
        parallelization="process",
    )

    remote_manager = RemoteAgentManager(
        client,
        agent_class=ResourceRequest(name="REINFORCEAgent"),
        train_env=ResourceRequest(name="gym_make", kwargs=dict(id="CartPole-v1")),
        fit_budget=FIT_BUDGET,
        init_kwargs=dict(gamma=0.99),
        eval_kwargs=dict(eval_horizon=200, n_simulations=20),
        n_fit=3,
        seed=10,
        agent_name="REINFORCE(remote)",
        parallelization="process",
        enable_tensorboard=True,
    )

    remote_manager.set_writer(
        idx=0,
        writer_fn=ResourceRequest(name="DefaultWriter"),
        writer_kwargs=dict(name="debug_reinforce_writer"),
    )

    # Optimize hyperparams of remote agent
    best_params = remote_manager.optimize_hyperparams(
        timeout=60, optuna_parallelization="process"
    )
    print(f"best params = {best_params}")

    # Test save/load
    fname1 = remote_manager.save()
    del remote_manager
    remote_manager = RemoteAgentManager.load(fname1)

    # Fit everything in parallel
    mmanagers = MultipleManagers(parallelization="thread")
    mmanagers.append(local_manager)
    mmanagers.append(remote_manager)
    mmanagers.run()

    # Fit remotely for a few more episodes
    remote_manager.fit(budget=100)

    # plot
    plot_writer_data(mmanagers.managers, tag="episode_rewards", show=False)
    evaluate_agents(mmanagers.managers, n_simulations=10, show=True)

    # Test some methods
    print([manager.eval_agents() for manager in mmanagers.managers])

    # # uncomment to clear output files
    # for manager in mmanagers.managers:
    #     manager.clear_handlers()
    #     manager.clear_output_dir()
