from datetime import datetime
import uuid
import hashlib
from typing import Optional, NamedTuple


# Default output directory used by the library.
RLBERRY_DEFAULT_DATA_DIR = "rlberry_data/"

# Temporary directory used by the library
RLBERRY_TEMP_DATA_DIR = "rlberry_data/temp/"


def get_timestamp_str():
    """
    Get a string containing current time stamp.
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    timestamp = date_time
    return timestamp


def get_readable_id(obj):
    """
    Create a more readable id than get_unique_id(),
    combining a timestamp to a 8-character hash.
    """
    long_id = get_unique_id(obj)
    timestamp = get_timestamp_str()
    short_id = f"{timestamp}_{long_id[:8]}"
    return short_id


def get_unique_id(obj):
    """
    Get a unique id for an obj. Use it in __init__ methods when necessary.
    """
    # id() is guaranteed to be unique among simultaneously existing objects (uses memory address).
    # uuid4() is an universal id, but there might be issues if called simultaneously in different processes.
    # This function combines id(), uuid4(), and a timestamp in a single ID, and hashes it.
    timestamp = datetime.timestamp(datetime.now())
    timestamp = str(timestamp).replace(".", "")
    str_id = timestamp + str(id(obj)) + uuid.uuid4().hex
    str_id = hashlib.md5(str_id.encode()).hexdigest()
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
