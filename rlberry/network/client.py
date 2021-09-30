import pprint
import socket
import json
from typing import List, Union
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

    def send(
            self,
            *messages: interface.Message,
            print_response: bool = False,
    ) -> Union[List[interface.Message], interface.Message]:
        returned_messages = []
        pp = pprint.PrettyPrinter(indent=4)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._host, self._port))
            for msg in messages:
                msg_bytes = serialize_message(msg)
                interface.send_data(s, msg_bytes)
                received_bytes = interface.receive_data(s)
                received_msg_dict = json.loads(received_bytes)
                if print_response:
                    pp.pprint(received_msg_dict)
                received_msg = interface.Message.from_dict(received_msg_dict)
                returned_messages.append(received_msg)

        if len(messages) == 1:
            return returned_messages[0]
        return returned_messages
