from rlberry.network.interface import ResourceItem
from rlberry.network.server import BerryServer
from rlberry.agents import ValueIterationAgent
from rlberry.agents.torch import REINFORCEAgent
from rlberry.envs import GridWorld, gym_make
from rlberry.utils.writers import DefaultWriter

if __name__ == '__main__':
    resources = dict(
        GridWorld=ResourceItem(
            obj=GridWorld,
            description='GridWorld constructor'
        ),
        gym_make=ResourceItem(
            obj=gym_make,
            description='gym_make'
        ),
        REINFORCEAgent=ResourceItem(
            obj=REINFORCEAgent,
            description='REINFORCEAgent'
        ),
        ValueIterationAgent=ResourceItem(
            obj=ValueIterationAgent,
            description='ValueIterationAgent constructor' + ValueIterationAgent.__doc__
        ),
        DefaultWriter=ResourceItem(
            obj=DefaultWriter,
            description='rlberry default writer'
        )
    )
    server = BerryServer(resources=resources, client_socket_timeout=10.0)
    server.start()
