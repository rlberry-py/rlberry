from typing import Any, Dict, Mapping, NamedTuple, Optional

REQUEST_PREFIX = 'ResourceRequest_'


class Message(NamedTuple):
    command: Optional[str] = None
    params: Optional[Mapping[str, Any]] = None
    data: Optional[Mapping[str, Any]] = None

    def to_dict(self):
        return self._asdict()

    @classmethod
    def create(cls, command=None, params=None, data=None):
        command = command or ''
        params = params or dict()
        data = data or dict()
        return cls(
            command=command,
            params=params,
            data=data
        )

    @classmethod
    def from_dict(cls, dict_message):
        return cls(**dict_message)


class ResourceItem(Dict):
    obj: Any
    description: str


Resources = Mapping[str, ResourceItem]


class ResourceRequest(Dict):
    name: str = ""
    kwargs: Optional[Mapping[str, Any]] = None
