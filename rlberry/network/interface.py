from typing import Any, Dict, Mapping, NamedTuple, Optional


REQUEST_PREFIX = 'ResourceRequest_'


class Command:
    NONE = 'NONE'
    ECHO = 'ECHO'
    CREATE_AGENT_STATS_INSTANCE = 'CREATE_AGENT_STATS_INSTANCE'
    FIT_AGENT_STATS = 'FIT_AGENT_STATS'
    EVAL_AGENT_STATS = 'EVAL_AGENT_STATS'
    LIST_RESOURCES = 'LIST_RESOURCES'


class BerryServerInfo:
    host: str
    port: int


class Message(NamedTuple):
    command: Optional[Command] = None
    params: Optional[Mapping[str, Any]] = None
    data: Optional[Mapping[str, Any]] = None
    info: Optional[Mapping[str, Any]] = None

    def to_dict(self):
        return self._asdict()

    @classmethod
    def create(cls, command=None, params=None, data=None, info=None):
        command = command or ''
        params = params or dict()
        data = data or dict()
        info = info or dict()
        return cls(
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
