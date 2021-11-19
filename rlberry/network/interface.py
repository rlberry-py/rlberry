import struct
from typing import Any, Dict, Mapping, NamedTuple, Optional


REQUEST_PREFIX = 'ResourceRequest_'


class Command:
    NONE = 'NONE'
    RAISE_EXCEPTION = 'RAISE_EXCEPTION'
    ECHO = 'ECHO'
    LIST_RESOURCES = 'LIST_RESOURCES'
    AGENT_MANAGER_CREATE_INSTANCE = 'AGENT_MANAGER_CREATE_INSTANCE'
    AGENT_MANAGER_FIT = 'AGENT_MANAGER_FIT'
    AGENT_MANAGER_EVAL = 'AGENT_MANAGER_EVAL'
    AGENT_MANAGER_CLEAR_OUTPUT_DIR = 'AGENT_MANAGER_CLEAR_OUTPUT_DIR'
    AGENT_MANAGER_CLEAR_HANDLERS = 'AGENT_MANAGER_CLEAR_HANDLERS'
    AGENT_MANAGER_SET_WRITER = 'AGENT_MANAGER_SET_WRITER'
    AGENT_MANAGER_OPTIMIZE_HYPERPARAMS = 'AGENT_MANAGER_OPTIMIZE_HYPERPARAMS'
    AGENT_MANAGER_GET_WRITER_DATA = 'AGENT_MANAGER_GET_WRITER_DATA'


class BerryServerInfo:
    host: str
    port: int


class Message(NamedTuple):
    message: Optional[str] = ''
    command: Optional[Command] = None
    params: Optional[Mapping[str, Any]] = None
    data: Optional[Mapping[str, Any]] = None
    info: Optional[Mapping[str, Any]] = None

    def to_dict(self):
        return self._asdict()

    @classmethod
    def create(
            cls,
            message: Optional[str] = '',
            command: Optional[Command] = None,
            params: Optional[Mapping[str, Any]] = None,
            data: Optional[Mapping[str, Any]] = None,
            info: Optional[Mapping[str, Any]] = None):
        command = command or ''
        params = params or dict()
        data = data or dict()
        info = info or dict()
        return cls(
            message=message,
            command=command,
            params=params,
            data=data,
            info=info,
        )

    @classmethod
    def from_dict(cls, dict_message):
        return cls(**dict_message)


class ResourceItem(Dict):
    obj: Any
    description: str


Resources = Mapping[str, ResourceItem]


class ResourceRequest(NamedTuple):
    name: str = ""
    kwargs: Optional[Mapping[str, Any]] = None


def next_power_of_two(x: int):
    return 1 << (x - 1).bit_length()


def send_data(socket, data):
    """
    adapted from: https://stackoverflow.com/a/63532988
    """
    print(f'[rlberry.network] sending {len(data)} bytes...')
    socket.sendall(struct.pack('>I', len(data)) + data)


def receive_data(socket):
    """
    adapted from: https://stackoverflow.com/a/63532988
    """
    data_size_packed = socket.recv(4)
    if not data_size_packed:
        return data_size_packed
    data_size = struct.unpack('>I', data_size_packed)[0]
    received_data = b""
    remaining_size = min(next_power_of_two(data_size), 4096)
    while remaining_size > 0:
        received_data += socket.recv(remaining_size)
        remaining_size = data_size - len(received_data)
        print(f'[rlberry.network] ... received {len(received_data)}/{data_size} bytes.')
    return received_data
