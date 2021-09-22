import pprint
import socket
import json
from rlberry.network import interface
from rlberry.network.utils import serialize_message


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

    def send(self, *messages: interface.Message):
        pp = pprint.PrettyPrinter(indent=4)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._host, self._port))
            for msg in messages:
                msg_bytes = serialize_message(msg)
                s.sendall(msg_bytes)
                received_bytes = s.recv(1024)
                pp.pprint(json.loads(received_bytes))
