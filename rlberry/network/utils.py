import json
from copy import deepcopy
from rlberry.network.resources import ResourceRequest
from typing import Any, Mapping


def json_serialize(obj: Mapping[str, Any]):
    obj_copy = deepcopy(obj)
    for key in obj:
        if isinstance(obj[key], ResourceRequest):
            val = obj_copy.pop(key)
            new_key = 'ResourceRequest_' + key
            obj_copy[new_key] = val
    obj = obj_copy

    def default(obj):
        return f"<<non-serializable: {type(obj).__qualname__}>>"
    return json.dumps(obj, default=default)
