from rlberry.network.resources import ResourceItem
from rlberry.network.server import BerryServer
from rlberry.agents import ValueIterationAgent
from rlberry.envs import GridWorld

resources = dict(
    GridWorld=ResourceItem(
        obj=GridWorld,
        description='GridWorld constructor'
    ),
    ValueIterationAgent=ResourceItem(
        obj=ValueIterationAgent,
        description='ValueIterationAgent constructor'
    ),
)

server = BerryServer(resources=resources)
server.start()
