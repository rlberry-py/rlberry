from rlberry.network.client import BerryClient
from rlberry.network.interface import ResourceRequest

from rlberry.agents import ValueIterationAgent as LocalValueIterationAgent
from rlberry.envs.finite import GridWorld as LocalGridWorld

from rlberry.manager.agent_manager import AgentManager
from rlberry.manager.multiple_managers import MultipleManagers
from rlberry.manager.remote_agent_manager import RemoteAgentManager


client1 = BerryClient(port=8001)
client2 = client1
# client2 = BerryClient(port=8004)


local_stats = AgentManager(
    agent_class=LocalValueIterationAgent,
    train_env=(LocalGridWorld, dict(nrows=35)),
    fit_budget=100,
    init_kwargs=dict(gamma=0.99),
    eval_kwargs=dict(eval_horizon=100, n_simulations=20),
    n_fit=2,
    seed=10
)


remote_stats1 = RemoteAgentManager(
    client1,
    agent_class=ResourceRequest(name='REINFORCEAgent'),
    train_env=ResourceRequest(name='gym_make', kwargs=dict(id='CartPole-v0')),
    fit_budget=100,
    init_kwargs=dict(gamma=0.95),
    eval_kwargs=dict(eval_horizon=100, n_simulations=20),
    n_fit=2,
    seed=10
)

remote_stats2 = RemoteAgentManager(
    client2,
    agent_class=ResourceRequest(name='A2CAgent'),
    train_env=ResourceRequest(name='gym_make', kwargs=dict(id='CartPole-v0')),
    fit_budget=100,
    init_kwargs=dict(gamma=0.95),
    eval_kwargs=dict(eval_horizon=100, n_simulations=20),
    n_fit=2,
    seed=10
)


remote_stats1.set_writer(
    idx=0,
    writer_fn=ResourceRequest(name='DefaultWriter'),
    writer_kwargs=dict(name='debug_reinforce_writer')
)

# Fit everything in parallel
mmanagers = MultipleManagers()
mmanagers.append(local_stats)
mmanagers.append(remote_stats1)
mmanagers.append(remote_stats2)
mmanagers.run()

# Test some methods
print([stats.eval_agents() for stats in mmanagers.managers])
local_stats.clear_output_dir()


best_params1 = remote_stats1.optimize_hyperparams(timeout=10)
best_params2 = remote_stats2.optimize_hyperparams(timeout=10)

print(f'best params 1 = {best_params1}')
print(f'best params 2 = {best_params2}')

remote_stats1.clear_handlers()
remote_stats1.clear_output_dir()

remote_stats2.clear_handlers()
remote_stats2.clear_output_dir()
