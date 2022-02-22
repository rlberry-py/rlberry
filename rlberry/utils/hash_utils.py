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
        # pipe output to /dev/null for silence
        null = open("/dev/null", "w")
        subprocess.Popen("git", stdout=null, stderr=null)
        null.close()
        git_installed = True

    except OSError:
        git_installed = False
    if git_installed:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            .strip()
            .decode()
        )
    else:
        return rlberry.__version__
