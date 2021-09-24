from typing import Any, Dict, Mapping, NamedTuple, Optional


REQUEST_PREFIX = 'ResourceRequest_'


class Command:
    NONE = 'NONE'
    ECHO = 'ECHO'
    LIST_RESOURCES = 'LIST_RESOURCES'
    CREATE_AGENT_STATS_INSTANCE = 'CREATE_AGENT_STATS_INSTANCE'
    FIT_AGENT_STATS = 'FIT_AGENT_STATS'
    EVAL_AGENT_STATS = 'EVAL_AGENT_STATS'
    AGENT_STATS_CLEAR_OUTPUT_DIR = 'AGENT_STATS_CLEAR_OUTPUT_DIR'


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
    def create(cls, message='', command=None, params=None, data=None, info=None):
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
