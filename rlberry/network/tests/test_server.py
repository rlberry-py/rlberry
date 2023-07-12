import sys

import py
import pytest
from xprocess import ProcessStarter
import numpy as np

from rlberry.network.client import BerryClient
from rlberry.network import interface
from rlberry.network.interface import Message, ResourceRequest
from rlberry.manager.remote_agent_manager import RemoteAgentManager
from rlberry.manager.evaluation import evaluate_agents

server_name = "berry"


@pytest.fixture(autouse=True)
def start_server(xprocess):
    python_executable_full_path = sys.executable
    python_server_script_full_path = py.path.local(__file__).dirpath("conftest.py")

    class Starter(ProcessStarter):
        pattern = "completed"
        args = [python_executable_full_path, python_server_script_full_path]

    xprocess.ensure(server_name, Starter)
    yield
    xprocess.getinfo(server_name).terminate()


def test_client():
    port = 4242
    client = BerryClient(port=port)
    # Send params for AgentManager
    client.send(
        Message.create(
            command=interface.Command.AGENT_MANAGER_CREATE_INSTANCE,
            params=dict(
                agent_class=ResourceRequest(name="ValueIterationAgent"),
                train_env=ResourceRequest(name="GridWorld", kwargs=dict(nrows=3)),
                fit_budget=2,
                init_kwargs=dict(gamma=0.95),
                eval_kwargs=dict(eval_horizon=2, n_simulations=2),
                n_fit=2,
                seed=10,
            ),
            data=None,
        ),
        Message.create(
            command=interface.Command.LIST_RESOURCES, params=dict(), data=dict()
        ),
    )

    client.send(
        Message.create(
            command=interface.Command.NONE,
            params=dict(),
            data=dict(big_list=list(1.0 * np.arange(2**4))),
        ),
        print_response=True,
    )


def test_remote_manager():
    port = 4242
    client = BerryClient(port=port)
    remote_manager = RemoteAgentManager(
        client,
        agent_class=ResourceRequest(name="REINFORCEAgent"),
        train_env=ResourceRequest(name="gym_make", kwargs=dict(id="CartPole-v1")),
        fit_budget=10,
        init_kwargs=dict(gamma=0.99),
        eval_kwargs=dict(eval_horizon=2, n_simulations=2),
        n_fit=2,
        agent_name="REINFORCE(remote)",
    )
    remote_manager.set_writer(
        idx=0,
        writer_fn=ResourceRequest(name="DefaultWriter"),
        writer_kwargs=dict(name="debug_reinforce_writer"),
    )

    # Optimize hyperparams of remote agent
    best_params = remote_manager.optimize_hyperparams(timeout=1)
    print(f"best params = {best_params}")

    fname1 = remote_manager.save()
    del remote_manager
    remote_manager = RemoteAgentManager.load(fname1)
    remote_manager.fit(3)
    evaluate_agents([remote_manager], n_simulations=2, show=False)
