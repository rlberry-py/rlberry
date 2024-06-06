from rlberry.manager import with_venv


@with_venv(import_libs=["rlberry"])
def run_sb():
    import rlberry  # noqa


def test_venv():
    run_venv_xp()
