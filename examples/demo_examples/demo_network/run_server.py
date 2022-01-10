""" 
 ===================== 
 Demo: run_server 
 =====================
"""
from rlberry.network.interface import ResourceItem
from rlberry.network.server import BerryServer
from rlberry.agents import ValueIterationAgent
from rlberry.agents.torch import REINFORCEAgent, A2CAgent
from rlberry.envs import GridWorld, gym_make
from rlberry.utils.writers import DefaultWriter

if __name__ == '__main__':
    port = int(input("Select server port: "))
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
        A2CAgent=ResourceItem(
            obj=A2CAgent,
            description='A2CAgent'
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
    server = BerryServer(resources=resources, port=port, client_socket_timeout=120.0)
    server.start()
