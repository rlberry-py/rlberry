import re
import inspect
from textwrap import dedent
import os
import nox
import subprocess
import tempfile


temp_dir = tempfile.TemporaryDirectory()


def with_venv(import_libs, python_ver=None):
    def wrap(func):
        fun_source = inspect.getsource(func)
        name = fun_source.split("\n")[1]  # skip decorator

        m = re.search(
            "(?<= )\w+", name
        )  # isolate the name of function to use as script name

        source = "\n" + dedent("\n".join(fun_source.split("\n")[2:]))

        filename = os.path.join(temp_dir.name, m.group(0) + ".py")

        with open(filename, "w") as f:
            f.write(dedent(source))

        @nox.session(name=m.group(0), python=python_ver)
        def myscript(session):
            for lib in import_libs:
                session.install(lib, silent=True)
            session.run("python", filename, silent=False)

        return myscript

    return wrap


def run_xp(reuse=True, venv_dir_name="rlberry_venvs", verbose=False):
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    run_file = module.__file__

    args = []
    if verbose:
        args.append("--verbose")

    if reuse:
        args.append("-r")
    subprocess.run(
        ["nox", "-f", run_file]
        + args
        + ["--envdir", os.path.join(os.path.dirname(run_file), venv_dir_name)]
    )
    temp_dir.cleanup()
