import concurrent.futures
import logging
import multiprocessing
import socket
import json
from copy import deepcopy
from rlberry.network import interface
from rlberry.network.utils import serialize_message
from rlberry.envs import gym_make
from rlberry.stats import AgentStats
from typing import Optional


logger = logging.getLogger(__name__)


class ClientHandler:
    def __init__(self, client_socket, client_address, resources, timeout):
        self._socket = client_socket
        self._address = client_address
        self._resources = resources
        self._timeout = timeout

    def _process_message(self, message: interface.Message):
        """Replace resource requests in 'message' by available resources."""
        message = message.to_dict()
        processed_message = deepcopy(message)
        for entry in ['params', 'data']:
            for key in message[entry]:
                if key.startswith(interface.REQUEST_PREFIX):
                    new_key = key[len(interface.REQUEST_PREFIX):]
                    resource_name = message[entry][key]['name']
                    try:
                        resource_kwargs = message[entry][key]['kwargs']
                    except KeyError:
                        resource_kwargs = None
                    if resource_name in self._resources:
                        processed_message[entry].pop(key)
                        if resource_kwargs:
                            processed_message[entry][new_key] = (self._resources[resource_name]['obj'], resource_kwargs)
                        else:
                            processed_message[entry][new_key] = self._resources[resource_name]['obj']
        return interface.Message.from_dict(processed_message)

    def run(self):
        if self._timeout:
            self._socket.settimeout(self._timeout)
        with self._socket:
            print(f'\n<client process> Handling client @ {self._address}')
            while True:
                message_bytes = self._socket.recv(1024)
                if not message_bytes:
                    break
                # process bytes
                message = interface.Message.from_dict(json.loads(message_bytes))
                message = self._process_message(message)
                print(message)
                # Handle message ...
                # Send confirmation
                self._socket.sendall(serialize_message(interface.Message.create(command='received_message')))
            print(f'<client process> Finished client @ {self._address}')


class BerryServer():
    """
    rlberry server

    Parameters
    ----------
    host :
        hostname, IP address or empty string.
    port : int
        Integer from 1 to 65535.
    backlog : int
        Number of unnaccepted connections allowed before refusing new connections.
    resources : Resources
        List of resources that can be requested by client.
    client_session_timeout : float
        Time (in seconds) that client process is kept alive.
    """
    def __init__(
        self,
        host='127.0.0.1',
        port: int = 65432,
        backlog: int = 5,
        resources: Optional[interface.Resources] = None,
        client_session_timeout: Optional[float] = None,
    ) -> None:
        assert port >= 1 and port <= 65535
        self._host = host
        self._port = port
        self._backlog = backlog

        self._resources = resources
        self._client_session_timeout = client_session_timeout

        # Define basic resources
        if resources is None:
            self._resources = dict(
                gym_make=interface.ResourceItem(
                    obj=gym_make,
                    description='gym_make'),
            )
        else:
            for key, val in resources.items():
                if set(val.keys()) != set(['obj', 'description']):
                    raise ValueError(
                        "resources items must be a dictionary with keys ['obj', 'description']."
                        f" Received: {list(val.keys())}")

    def start(self):
        print(f'\n\nStarting BerryServer @ (host, port) = ({self._host}, {self._port}).\n\n')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self._host, self._port))
            s.listen(self._backlog)
            with concurrent.futures.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn')) as executor:
                while True:
                    print(f'<main process> BerryServer({self._host}, {self._port}): waiting for connection...')
                    client_socket, client_address = s.accept()   # wait for connection
                    client_handler = ClientHandler(
                        client_socket,
                        client_address,
                        self._resources,
                        self._client_session_timeout)
                    executor.submit(client_handler.run)
                    # with conn:
                    #     print(f'BerryServer({self._host}, {self._port}): new connection by {addr}')
                    #     while True:
                    #         data = conn.recv(1024)
                    #         if not data:
                    #             break
                    #         try:
                    #             data = self._process_received_data(data)
                    #         except Exception as ex:
                    #             print(f'Error: {str(ex)}')
                    #         print(f'\n Received data: {data}, of type {type(data)}')
                    #         if data['command'] == 'agent_stats':
                    #             print('\n Initializing AgentStats...')
                    #             agent_stats = AgentStats(**data['params'])
                    #             agent_stats.fit()
                    #             output = Message.create(
                    #                 command=dict(),
                    #                 params=dict(),
                    #                 data=dict(eval_output=agent_stats.eval()))
                    #             conn.sendall(serialize_message(output))

                    #         else:
                    #             for rr in self._resources:
                    #                 print('==============================================')
                    #                 print(f'Resource: \n---------\n{rr}\n')
                    #                 print(f"Description: \n------------\n{self._resources[rr]['description']}\n\n")
                    #                 print('==============================================')
                    #             conn.sendall(serialize_message(Message.create(command='end')))
 

if __name__ == '__main__':
    server = BerryServer()
    server.start()
