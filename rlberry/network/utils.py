import json
from copy import deepcopy
from rlberry.network import interface


def serialize_message(message: interface.Message) -> bytes:
    message = message.to_dict()
    processes_msg = deepcopy(message)
    for entry in ['params', 'data']:
        for key in message[entry]:
            if isinstance(message[entry][key], interface.ResourceRequest):
                val = processes_msg[entry].pop(key)
                new_key = interface.REQUEST_PREFIX + key
                processes_msg[entry][new_key] = val
    message = processes_msg

    def default(obj):
        return f"<<non-serializable: {type(obj).__qualname__}>>"

    return str.encode(json.dumps(message, default=default))
