import subprocess
import os
import rlberry


def get_rlberry_version():
    """
    Returns
    -------

    the hash of the last rlberry git commit, a string.
    """

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            .strip()
            .decode()
        )
    except:
        return rlberry.__version__
