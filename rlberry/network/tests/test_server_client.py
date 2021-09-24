# import concurrent.futures
# import multiprocessing
# import rlberry.network.interface as interface
# from rlberry.network.client import BerryClient
# from rlberry.network.server import BerryServer
# from rlberry.network.interface import ResourceItem
# from rlberry.network.interface import Message, ResourceRequest
# from rlberry.agents import ValueIterationAgent
# from rlberry.envs import GridWorld


# PORT = 65001


# def run_server():
#     resources = dict(
#         GridWorld=ResourceItem(
#             obj=GridWorld,
#             description='GridWorld constructor'
#         ),
#         ValueIterationAgent=ResourceItem(
#             obj=ValueIterationAgent,
#             description='ValueIterationAgent constructor' + ValueIterationAgent.__doc__
#         ),
#     )
#     server = BerryServer(
#         port=PORT,
#         resources=resources,
#         client_socket_timeout=0.5,
#         terminate_after=2)
#     server.start()


# def run_client():
#     client = BerryClient(port=PORT)
#     # Send params for AgentStats
#     output = client.send(
#         Message.create(
#             command=interface.Command.CREATE_AGENT_STATS_INSTANCE,
#             params=dict(
#                 agent_class=ResourceRequest(name='ValueIterationAgent'),
#                 train_env=ResourceRequest(name='GridWorld', kwargs=dict(nrows=35)),
#                 fit_budget=100,
#                 init_kwargs=dict(gamma=0.95),
#                 eval_kwargs=dict(eval_horizon=100, n_simulations=20),
#                 n_fit=2,
#                 seed=10
#             ),
#             data=None,
#         ),
#         Message.create(
#             command=interface.Command.LIST_RESOURCES,
#             params=dict(),
#             data=dict()
#         )
#     )

#     client.send(
#         Message.create(
#             command=interface.Command.AGENT_STATS_CLEAR_OUTPUT_DIR,
#             params=dict(filename=output[0].info['filename'])
#         )
#     )


# def test_server_client():
#     futures = []
#     with concurrent.futures.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn')) as executor:
#         futures.append(executor.submit(run_server))
#         futures.append(executor.submit(run_client))


# if __name__ == '__main__':
#     test_server_client()
