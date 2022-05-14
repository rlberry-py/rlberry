import pytest

import os
from rlberry.network.interface import ResourceItem
from rlberry.network.server import BerryServer
from rlberry.envs import GridWorld
from rlberry.utils.writers import DefaultWriter
from xprocess import ProcessStarter
from rlberry.network.client import BerryClient
from rlberry.network.interface import ResourceRequest

from rlberry.agents import UCBVIAgent

from rlberry.manager.remote_agent_manager import RemoteAgentManager

import py
import sys


def fit_client():
    port = 4242

    client = BerryClient(port=port)

    remote_manager = RemoteAgentManager(
        client,
        agent_class=ResourceRequest(name="UCBVIAgent"),
        train_env=ResourceRequest(name="GridWorld", kwargs={}),
        fit_budget=5,
        n_fit=2,
        seed=10,
    )
    # Optimize hyperparams of remote agent
    best_params = remote_manager.optimize_hyperparams(timeout=2)
    print(f"best params = {best_params}")

    # Test save/load
    fname1 = remote_manager.save()
    del remote_manager
    remote_manager = RemoteAgentManager.load(fname1)

    remote_manager.fit(budget=5)


server_name = "run_server"


@pytest.fixture(autouse=True)
def start_server(xprocess):
    print("starting the server")

    python_executable_full_path = sys.executable
    python_server_script_full_path = py.path.local(__file__).dirpath("run_server.py")

    class Starter(ProcessStarter):
        pattern = "completed"
        port = 4242
        args = [python_executable_full_path, python_server_script_full_path]

    # ensure process is running and return its logfile
    xprocess.ensure(server_name, Starter)

    yield

    xprocess.getinfo(server_name).terminate()


def test_client():
    print("starting client test")
    fit_client()  # create a connection or url/port info to the server
