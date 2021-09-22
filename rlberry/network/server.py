import logging
import socket
import json
from copy import deepcopy
from rlberry.network.resources import Resources, ResourceItem
from typing import Optional
from rlberry.network.utils import json_serialize
from rlberry.envs import gym_make
from rlberry.stats import AgentStats


logger = logging.getLogger(__name__)


class BerryServer():
    """
    rlberry server

    Parameters
    ----------
    host :
        hostname, IP address or empty string.
    port : int
        Integer from 1-65535
    backlog : int
        Number of unnaccepted connections allowed before refusing new connections.
    resources : Resources
        List of resources that can be requested by client.
    """
    def __init__(
        self,
        host='127.0.0.1',
        port: int = 65432,
        backlog: int = 5,
        resources: Optional[Resources] = None,
    ) -> None:
        assert port >= 1 and port <= 65535
        self._host = host
        self._port = port
        self._backlog = backlog

        self._resources = resources

        # Define basic resources
        if resources is None:
            self._resources = dict(
                gym_make=ResourceItem(
                    obj=gym_make,
                    description='gym_make'),
            )
        else:
            for key, val in resources.items():
                if set(val.keys()) != set(['obj', 'description']):
                    raise ValueError(
                        "resources items must be a dictionary with keys ['obj', 'description']."
                        f" Received: {list(val.keys())}")

    def _process_received_data(self, received):
        data = json.loads(received)
        assert isinstance(data, dict)
        processed_data = deepcopy(data)

        request_prefix = 'ResourceRequest_'
        for key in data:
            if key.startswith(request_prefix):
                new_key = key[len(request_prefix):]
                resource_name = data[key]['name']
                try:
                    resource_kwargs = data[key]['kwargs']
                except KeyError:
                    resource_kwargs = None
                if resource_name in self._resources:
                    processed_data.pop(key)
                    if resource_kwargs:
                        processed_data[new_key] = (self._resources[resource_name]['obj'], resource_kwargs)
                    else:
                        processed_data[new_key] = self._resources[resource_name]['obj']
        return processed_data

    def start(self):
        print(f'Starting BerryServer @ (host, port) = ({self._host}, {self._port}).')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self._host, self._port))
            s.listen(self._backlog)
            while True:
                print(f'BerryServer({self._host}, {self._port}): waiting for connection...')
                conn, addr = s.accept()   # wait for connection
                with conn:
                    print(f'BerryServer({self._host}, {self._port}): new connection by {addr}')
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        try:
                            data = self._process_received_data(data)
                        except Exception as ex:
                            print(f'Error: {str(ex)}')
                        # print(f'\n Received data: {data}, of type {type(data)}')
                        print('\n Initializing AgentStats...')
                        agent_stats = AgentStats(**data)
                        agent_stats.fit()
                        eval_output = dict(eval_output=agent_stats.eval())
                        eval_output = json_serialize(eval_output)
                        eval_output = str.encode(eval_output)
                        conn.sendall(eval_output)


if __name__ == '__main__':
    server = BerryServer()
    server.start()
