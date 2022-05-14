from rlberry.network.interface import ResourceItem
from rlberry.network.server import BerryServer
from rlberry.agents import UCBVIAgent
from rlberry.envs import GridWorld
from rlberry.utils.writers import DefaultWriter
import os
import sys


def print_err(s):
    sys.stderr.write(s)
    sys.stderr.flush()


if __name__ == "__main__":
    port = 4242
    resources = dict(
        GridWorld=ResourceItem(obj=GridWorld, description="GridWorld constructor"),
        REINFORCEAgent=ResourceItem(obj=UCBVIAgent, description="UCBVIAgent"),
        DefaultWriter=ResourceItem(
            obj=DefaultWriter, description="rlberry default writer"
        ),
    )
    server = BerryServer(resources=resources, port=port, client_socket_timeout=120.0)
    os.system("touch ha")
    server.start()
