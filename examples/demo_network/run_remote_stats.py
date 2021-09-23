from rlberry.network.client import BerryClient
from rlberry.network.interface import ResourceRequest

from rlberry.stats.remote_agent_stats import RemoteAgentStats


client = BerryClient()


remote_stats = RemoteAgentStats(
    client,
    agent_class=ResourceRequest(name='ValueIterationAgent'),
    train_env=ResourceRequest(name='GridWorld', kwargs=dict(nrows=35)),
    fit_budget=100,
    init_kwargs=dict(gamma=0.95),
    eval_kwargs=dict(eval_horizon=100, n_simulations=20),
    n_fit=2,
    seed=10
)


remote_stats.fit()

print(remote_stats.eval())
