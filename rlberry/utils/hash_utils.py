import hashlib
import pickle
import subprocess
import os


def get_pickle_md5(object):
    """
    Get the md5 hash of an object

    Parameters
    ----------

    object : a python object
        object to hash. Typically used with an agent class or AgentManager instance.

    Returns
    -------

    the md5 hash of the object, a string.

    """
    return hashlib.md5(pickle.dumps(object)).hexdigest()


def get_git_hash():
    """
    Returns
    -------

    the hash of the last rlberry git commit, a string.
    """
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__))
        )
        .strip()
        .decode()
    )
