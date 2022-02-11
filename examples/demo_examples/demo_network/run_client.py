""" 
 ===================== 
 Demo: run_client 
 =====================
"""
from rlberry.network.client import BerryClient
from rlberry.network import interface
from rlberry.network.interface import Message, ResourceRequest


port = int(input("Select server port: "))
client = BerryClient(port=port)
# Send params for AgentManager
client.send(
    Message.create(
        command=interface.Command.AGENT_MANAGER_CREATE_INSTANCE,
        params=dict(
            agent_class=ResourceRequest(name="ValueIterationAgent"),
            train_env=ResourceRequest(name="GridWorld", kwargs=dict(nrows=35)),
            fit_budget=100,
            init_kwargs=dict(gamma=0.95),
            eval_kwargs=dict(eval_horizon=100, n_simulations=20),
            n_fit=2,
            seed=10,
        ),
        data=None,
    ),
    Message.create(
        command=interface.Command.LIST_RESOURCES, params=dict(), data=dict()
    ),
    print_response=True,
)

import numpy as np

client.send(
    Message.create(
        command=interface.Command.NONE,
        params=dict(),
        data=dict(big_list=list(1.0 * np.arange(2**8))),
    ),
    print_response=True,
)
