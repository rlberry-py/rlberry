import importlib
from typing import Callable


def load(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)
