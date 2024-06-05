import re
import inspect
from textwrap import dedent
import os
import subprocess

try:
    import nox

    NOX_INSTALLED = True
except:
    NOX_INSTALLED = False


temp_dir = "rlberry_venvs"


def __func_to_script(func):
    fun_source = inspect.getsource(func)
    name = fun_source.split("\n")[1]  # skip decorator

    m = re.search(
        "(?<= )\w+", name
    )  # isolate the name of function to use as script name

    source = "\n" + dedent("\n".join(fun_source.split("\n")[2:]))
    source = dedent(source)

    filename = os.path.join(temp_dir, m.group(0) + ".py")

    with open(filename, "w") as f:
        f.write(source)
    return filename


def with_venv(import_libs=None, requirements=None, python_ver=None, verbose=False):
    def wrap(func):
        assert (
            NOX_INSTALLED
        ), "module not found: nox. nox must be installed to use rlberry's venv tools"
        if not (os.path.isdir(temp_dir)):
            os.mkdir(temp_dir)

        filename = __func_to_script(func)

        assert (
            import_libs or requirement or pyproject_toml
        ), "At least one of import_libs or requirements must be not None"

        @nox.session(name=filename.split(".")[0], reuse_venv=True, python=python_ver)
        def myscript(session):
            if requirements:
                session.install("-r", requirements)
            else:
                for lib in import_libs:
                    session.install(lib, silent=not (verbose))
            with open(filename.split(".")[0] + "_requirements.txt", "w") as f:
                deps = session.run("python", "-m", "pip", "freeze", stdout=f)
            print(
                "A requirements.txt containing the frozen dependencies from the virtual environment."
            )
            session.run("python", filename, silent=False)

        return myscript

    return wrap


def run_venv_xp(venv_dir_name="rlberry_venvs", verbose=False):
    assert NOX_INSTALLED
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    run_file = module.__file__
    args = []
    if verbose:
        args.append("--verbose")

    # reuse the virtual environments
    args.append("-r")

    subprocess.run(
        ["nox", "-f", run_file]
        + args
        + ["--envdir", os.path.join(os.path.dirname(run_file), venv_dir_name)],
        check=True,
    )
