# content of conftest.py
# This file is used to spawn a server to connect to in the tests from test_server.py

import multiprocessing

from rlberry.network.interface import ResourceItem
from rlberry.network.server import BerryServer
from rlberry.agents import ValueIterationAgent
from rlberry.agents.torch import REINFORCEAgent
from rlberry.envs import GridWorld, gym_make
from rlberry.utils.writers import DefaultWriter

import sys


def print_err(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def server(port):
    # definition of server
    resources = dict(
        GridWorld=ResourceItem(obj=GridWorld, description="GridWorld constructor"),
        gym_make=ResourceItem(obj=gym_make, description="gym_make"),
        REINFORCEAgent=ResourceItem(obj=REINFORCEAgent, description="REINFORCEAgent"),
        ValueIterationAgent=ResourceItem(
            obj=ValueIterationAgent,
            description="ValueIterationAgent constructor" + ValueIterationAgent.__doc__,
        ),
        DefaultWriter=ResourceItem(
            obj=DefaultWriter, description="rlberry default writer"
        ),
    )
    server = BerryServer(resources=resources, port=port, client_socket_timeout=120.0)
    server.start()


if __name__ == "__main__":
    default_port = 4242
    p = multiprocessing.Process(target=server, args=(default_port,))
    p.start()
    print_err("Server startup completed!")
