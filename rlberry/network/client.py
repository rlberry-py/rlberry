import socket
import json
from typing import Any, Mapping
from rlberry.network.utils import json_serialize


class BerryClient():
    """
    rlberry client

    Parameters
    ----------
    host :
        hostname, IP address or empty string.
    port : int
        Integer from 1-65535
    """
    def __init__(
        self,
        host='127.0.0.1',
        port: int = 65432,
    ) -> None:
        assert port >= 1 and port <= 65535
        self._host = host
        self._port = port

    def send(self, obj: Mapping[str, Any]):
        data = json_serialize(obj)
        data = str.encode(data)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._host, self._port))
            s.sendall(data)
            data = s.recv(1024)
        print(json.loads(data))
