import concurrent.futures
import logging
import multiprocessing
import socket
import json
from copy import deepcopy
from rlberry.network import interface
from rlberry.network.utils import serialize_message
from rlberry.envs import gym_make
from rlberry.manager import AgentManager
from typing import Optional


logger = logging.getLogger(__name__)


class ClientHandler:
    def __init__(self, client_socket, client_address, resources):
        self._socket = client_socket
        self._address = client_address
        self._resources = resources

    def _process_message(self, message: interface.Message):
        """Replace resource requests in 'message' by available resources."""
        message = message.to_dict()
        processed_message = deepcopy(message)
        for entry in ['params', 'data', 'info']:
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

    def _execute_message(self, message: interface.Message):
        """Execute command in message and send response."""
        response = interface.Message.create(command=interface.Command.ECHO)
        try:
            # LIST_RESOURCES
            if message.command == interface.Command.LIST_RESOURCES:
                info = {}
                for rr in self._resources:
                    info[rr] = self._resources[rr]['description']
                response = interface.Message.create(info=info)
            # CREATE_agent_manager_INSTANCE
            elif message.command == interface.Command.CREATE_agent_manager_INSTANCE:
                agent_manager = AgentManager(**message.params)
                output_dir = 'client_data' / agent_manager.output_dir
                agent_manager.set_output_dir(output_dir)
                filename = str(agent_manager.save())
                response = interface.Message.create(info=dict(filename=filename))
                del agent_manager
            # FIT_agent_manager
            elif message.command == interface.Command.FIT_agent_manager:
                filename = message.params['filename']
                agent_manager = AgentManager.load(filename)
                agent_manager.fit()
                agent_manager.save()
                response = interface.Message.create(command=interface.Command.ECHO)
                del agent_manager
            # EVAL_agent_manager
            elif message.command == interface.Command.EVAL_agent_manager:
                filename = message.params['filename']
                agent_manager = AgentManager.load(filename)
                eval_output = agent_manager.eval()
                # agent_manager.save()  # eval does not change the state of agent stats
                response = interface.Message.create(data=dict(output=eval_output))
                del agent_manager
            # agent_manager_CLEAR_OUTPUT_DIR
            elif message.command == interface.Command.agent_manager_CLEAR_OUTPUT_DIR:
                filename = message.params['filename']
                agent_manager = AgentManager.load(filename)
                agent_manager.clear_output_dir()
                response = interface.Message.create(message=f'Cleared output: {agent_manager.output_dir}')
                del agent_manager
            # Send response
            self._socket.sendall(serialize_message(response))
        except Exception as ex:
            response = interface.Message.create(info=dict(ERROR='Exception: ' + str(ex)))
            self._socket.sendall(serialize_message(response))
            return 1
        return 0

    def run(self):
        with self._socket:
            try:
                print(f'\n<server: client process> Handling client @ {self._address}')
                while True:
                    message_bytes = self._socket.recv(1024)
                    if not message_bytes:
                        break
                    # process bytes
                    message = interface.Message.from_dict(json.loads(message_bytes))
                    message = self._process_message(message)
                    print(f'<server: client process> Received message: \n{message}')
                    # execute message commands and send back a response
                    self._execute_message(message)
            except Exception as ex:
                print(f'<server: client process> [ERROR]: {ex}')
            finally:
                print(f'<server: client process> Finished client @ {self._address}')


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
    client_socket_timeout : float, default: 120
        Timeout (in seconds) for client socket operations.
    terminate_after : int
        Number of received client sockets after which to terminate the server. If None,
        does not terminate.
    """
    def __init__(
        self,
        host='127.0.0.1',
        port: int = 65432,
        backlog: int = 5,
        resources: Optional[interface.Resources] = None,
        client_socket_timeout: float = 120.0,
        terminate_after: Optional[int] = None,
    ) -> None:
        assert port >= 1 and port <= 65535
        self._host = host
        self._port = port
        self._backlog = backlog

        self._resources = resources
        self._client_socket_timeout = client_socket_timeout
        self._terminate_after = terminate_after
        self._client_socket_counter = 0

        # Define basic resources
        if resources is None:
            self._resources = dict(
                gym_make=interface.ResourceItem(
                    obj=gym_make,
                    description='gym_make'),
            )
        else:
            for _, val in resources.items():
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
                futures = []
                while True:
                    print(f'<server: main process> BerryServer({self._host}, {self._port}): waiting for connection...')
                    client_socket, client_address = s.accept()   # wait for connection
                    client_socket.settimeout(self._client_socket_timeout)
                    self._client_socket_counter += 1
                    client_handler = ClientHandler(
                        client_socket,
                        client_address,
                        self._resources)
                    futures.append(executor.submit(client_handler.run))
                    if self._terminate_after and self._client_socket_counter >= self._terminate_after:
                        print('<server: main process> Terminating server (main process): '
                              'reached max number of client sockets.')
                        break


if __name__ == '__main__':
    server = BerryServer()
    server.start()
