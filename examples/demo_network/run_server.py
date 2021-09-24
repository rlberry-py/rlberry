from rlberry.network.interface import ResourceItem
from rlberry.network.server import BerryServer
from rlberry.agents import ValueIterationAgent
from rlberry.envs import GridWorld


if __name__ == '__main__':
    resources = dict(
        GridWorld=ResourceItem(
            obj=GridWorld,
            description='GridWorld constructor'
        ),
        ValueIterationAgent=ResourceItem(
            obj=ValueIterationAgent,
            description='ValueIterationAgent constructor' + ValueIterationAgent.__doc__
        ),
    )

    server = BerryServer(resources=resources, client_socket_timeout=None)
    server.start()
