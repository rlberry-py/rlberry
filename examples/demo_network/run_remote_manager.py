from rlberry.network.client import BerryClient
from rlberry.network.interface import ResourceRequest

from rlberry.agents import ValueIterationAgent as LocalValueIterationAgent
from rlberry.envs.finite import GridWorld as LocalGridWorld

from rlberry.manager.agent_manager import AgentManager
from rlberry.manager.multiple_managers import MultipleManagers
from rlberry.manager.remote_agent_manager import RemoteAgentManager

client = BerryClient()


local_stats = AgentManager(
    agent_class=LocalValueIterationAgent,
    train_env=(LocalGridWorld, dict(nrows=35)),
    fit_budget=100,
    init_kwargs=dict(gamma=0.99),
    eval_kwargs=dict(eval_horizon=100, n_simulations=20),
    n_fit=2,
    seed=10
)


remote_stats = RemoteAgentManager(
    client,
    agent_class=ResourceRequest(name='ValueIterationAgent'),
    train_env=ResourceRequest(name='GridWorld', kwargs=dict(nrows=35)),
    fit_budget=100,
    init_kwargs=dict(gamma=0.95),
    eval_kwargs=dict(eval_horizon=100, n_simulations=20),
    n_fit=2,
    seed=10
)

mmanagers = MultipleManagers()
mmanagers.append(local_stats)
mmanagers.append(remote_stats)

mmanagers.run()

print([stats.eval() for stats in mmanagers.managers])
