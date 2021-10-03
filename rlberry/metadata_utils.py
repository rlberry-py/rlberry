import uuid
import hashlib
from typing import Optional, NamedTuple


# Default output directory used by the library.
RLBERRY_DEFAULT_DATA_DIR = 'rlberry_data/'

# Temporary directory used by the library
RLBERRY_TEMP_DATA_DIR = 'rlberry_data/temp/'


def get_unique_id(obj):
    """
    Get a unique id for an obj. Use it in __init__ methods when necessary.
    """
    # id() is guaranteed to be unique among simultaneously existing objects (uses memory address).
    # uuid4() is an universal id, but there might be issues if called simultaneously in different processes.
    # This function combines both in a single ID, and hashes it.
    str_id = str(id(obj)) + uuid.uuid4().hex
    str_id = hashlib.sha1(str_id.encode("utf-8")).hexdigest()
    return str_id


class ExecutionMetadata(NamedTuple):
    """
    Metadata for objects handled by rlberry.

    Attributes
    ----------
    obj_worker_id : int, default: -1
        If given, must be >= 0, and inform the worker id (thread or process) where the
        object was created. It is not necessarity unique across all the workers launched by
        rlberry, it is mainly for debug purposes.
    obj_info : dict, default: None
        Extra info about the object.
    """
    obj_worker_id: int = -1
    obj_info: Optional[dict] = None
