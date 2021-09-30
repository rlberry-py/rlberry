from rlberry.envs.gym_make import gym_make
from rlberry.network.client import BerryClient
from rlberry.network.interface import ResourceRequest

from rlberry.agents.torch import REINFORCEAgent

from rlberry.manager.agent_manager import AgentManager
from rlberry.manager.multiple_managers import MultipleManagers
from rlberry.manager.remote_agent_manager import RemoteAgentManager
from rlberry.manager.evaluation import evaluate_agents, plot_writer_data


if __name__ == '__main__':
    port = int(input("Select server port: "))
    client = BerryClient(port=port)

    FIT_BUDGET = 1000

    local_manager = AgentManager(
        agent_class=REINFORCEAgent,
        train_env=(gym_make, dict(id='CartPole-v0')),
        fit_budget=FIT_BUDGET,
        init_kwargs=dict(gamma=0.99),
        eval_kwargs=dict(eval_horizon=200, n_simulations=20),
        n_fit=2,
        seed=10,
        agent_name='REINFORCE(local)',
        parallelization='process'
    )

    remote_manager = RemoteAgentManager(
        client,
        agent_class=ResourceRequest(name='REINFORCEAgent'),
        train_env=ResourceRequest(name='gym_make', kwargs=dict(id='CartPole-v0')),
        fit_budget=FIT_BUDGET,
        init_kwargs=dict(gamma=0.99),
        eval_kwargs=dict(eval_horizon=200, n_simulations=20),
        n_fit=2,
        seed=10,
        agent_name='REINFORCE(remote)',
        parallelization='process'
    )

    remote_manager.set_writer(
        idx=0,
        writer_fn=ResourceRequest(name='DefaultWriter'),
        writer_kwargs=dict(name='debug_reinforce_writer')
    )

    # Optimize hyperparams of remote agent
    best_params = remote_manager.optimize_hyperparams(timeout=120, optuna_parallelization='process')
    print(f'best params = {best_params}')

    # Test save/load
    fname1 = remote_manager.save()
    del remote_manager
    remote_manager = RemoteAgentManager.load(fname1)

    # Fit everything in parallel
    mmanagers = MultipleManagers()
    mmanagers.append(local_manager)
    mmanagers.append(remote_manager)
    mmanagers.run()

    # plot
    plot_writer_data(mmanagers.managers, tag='episode_rewards', show=False)
    evaluate_agents(mmanagers.managers, n_simulations=10, show=True)

    # Test some methods
    print([manager.eval_agents() for manager in mmanagers.managers])

    for manager in mmanagers.managers:
        manager.clear_handlers()
        manager.clear_output_dir()
