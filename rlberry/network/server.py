import concurrent.futures
import logging
import multiprocessing
import socket
import json
import rlberry.network.server_utils as server_utils
from rlberry.network import interface
from rlberry.network.utils import apply_fn_to_tree, map_request_to_obj, serialize_message
from rlberry.envs import gym_make
from typing import Optional


logger = logging.getLogger(__name__)


class ClientHandler:
    def __init__(self, client_socket, client_address, resources, timeout):
        self._socket = client_socket
        self._address = client_address
        self._resources = resources
        self._logger = logging.getLogger('ClientHandler')
        self._timeout = timeout

    def _process_message(self, message: interface.Message):
        """Replace resource requests in 'message' by available resources."""
        message = message.to_dict()
        message = apply_fn_to_tree(
            lambda key, val: map_request_to_obj(key, val, self._resources), message, apply_to_nodes=True
        )
        return interface.Message.from_dict(message)

    def _execute_message(self, message: interface.Message):
        """Execute command in message and send response."""
        self._socket.settimeout(self._timeout)
        try:
            # Execute commands
            response = server_utils.execute_message(message, self._resources)
            # Send response
            interface.send_data(self._socket, serialize_message(response))
        except Exception as ex:
            response = interface.Message.create(
                command=interface.Command.RAISE_EXCEPTION,
                message=str(ex))
            interface.send_data(self._socket, serialize_message(response))
            return 1
        return 0

    def run(self):
        with self._socket:
            try:
                while True:
                    print(f'\n<server: client process> Handling client @ {self._address}')
                    self._socket.settimeout(self._timeout)
                    message_bytes = interface.receive_data(self._socket)
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
                self._logger.exception(ex)
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
                    self._client_socket_counter += 1
                    client_handler = ClientHandler(
                        client_socket,
                        client_address,
                        self._resources,
                        self._client_socket_timeout)
                    print(f'<server: main process> BerryServer({self._host}, {self._port}): '
                          f'new client @ {client_address}')
                    futures.append(executor.submit(client_handler.run))
                    if self._terminate_after and self._client_socket_counter >= self._terminate_after:
                        print('<server: main process> Terminating server (main process): '
                              'reached max number of client sockets.')
                        break


if __name__ == '__main__':
    server = BerryServer()
    server.start()
