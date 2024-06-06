from rlberry.manager import with_venv, run_venv_xp


@with_venv(import_libs=["tqdm"])
def run_tqdm():
    from tqdm import tqdm  # noqa


def test_venv():
    run_venv_xp()
