import json
from copy import deepcopy
from rlberry.network import interface
from typing import Any, Callable, Mapping, Optional, Tuple, Union


Tree = Union[Any, Tuple, Mapping[Any, 'Tree']]


def apply_fn_to_tree(
        fn: Callable[[Any, Any], Tuple[Any, Any]],
        tree: Tree,
        is_leaf: Optional[Callable[[Any], Any]] = None,
        apply_to_nodes: Optional[bool] = False):
    """
    new_key, new_val = fn(key, my_dict[key])
    """
    is_leaf = is_leaf or (lambda x: not isinstance(x, Mapping) and not isinstance(x, Tuple))
    if is_leaf(tree):
        return deepcopy(tree)
    if isinstance(tree, Mapping):
        new_tree = dict()
        keys = list(tree.keys())
        for key in keys:
            new_tree[key] = tree[key]
            if apply_to_nodes or is_leaf(tree[key]):
                new_key, new_val = fn(key, tree[key])
                new_tree.pop(key)
                new_tree[new_key] = new_val
        return {key: apply_fn_to_tree(
            fn, val, is_leaf, apply_to_nodes) for (key, val) in new_tree.items()}
    elif isinstance(tree, Tuple):
        return tuple([apply_fn_to_tree(fn, val, is_leaf, apply_to_nodes) for val in tree])
    else:
        raise RuntimeError('Tree is not a Mapping or Tuple.')


def _map_resource_request_to_dict(key, val):
    if isinstance(val, interface.ResourceRequest):
        assert isinstance(key, str)
        new_key = interface.REQUEST_PREFIX + key
        new_val = val._asdict()
        return new_key, new_val
    return key, val


def map_request_to_obj(key, val, resources: interface.Resources):
    if key.startswith(interface.REQUEST_PREFIX):
        new_key = key[len(interface.REQUEST_PREFIX):]
        resource_name = val['name']
        try:
            resource_kwargs = val['kwargs']
        except KeyError:
            resource_kwargs = None
        if resource_name in resources:
            if resource_kwargs:
                new_val = (resources[resource_name]['obj'], resource_kwargs)
            else:
                new_val = resources[resource_name]['obj']
            return new_key, new_val
        else:
            raise RuntimeError(f'Unavailable requested resource: {resource_name}')
    else:
        return key, val


def serialize_message(message: interface.Message) -> bytes:
    message = message.to_dict()
    message = apply_fn_to_tree(_map_resource_request_to_dict, message, apply_to_nodes=True)

    def default(obj):
        return f"<<non-serializable: {type(obj).__qualname__}>>"

    return str.encode(json.dumps(message, default=default))