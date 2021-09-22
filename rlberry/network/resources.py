from typing import Any, Mapping, Dict, Optional


class ResourceItem(Dict):
    obj: Any
    description: str


Resources = Mapping[str, ResourceItem]


class ResourceRequest(Dict):
    name: str = ""
    kwargs: Optional[Mapping[str, Any]] = None
