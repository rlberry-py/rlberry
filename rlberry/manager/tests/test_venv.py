import pytest
import sys
from rlberry.manager import with_venv, run_venv_xp


@with_venv(import_libs=["tqdm"], verbose=True)
def run_tqdm():
    from tqdm import tqdm  # noqa


@pytest.mark.skip(sys.platform == "darwin", reason="bug with MacOS_14")
def test_venv():
    run_venv_xp(verbose=True)
