import re
import inspect
from textwrap import dedent
import os
import subprocess
import tempfile
import shutil
import glob

try:
    import nox

    NOX_INSTALLED = True
except:
    NOX_INSTALLED = False


temp_dir = tempfile.mkdtemp()

template_script_guix = """#!/usr/bin/env -S guix time-machine --channels=channels.scm -- shell -CFN --preserve='(^DISPLAY$|^XAUTHORITY$)' --expose=${HOME}/.Xauthority -m manifest.scm -- bash

export LD_LIBRARY_PATH=/lib

"""

nonguix_channel = """
(cons* (channel
        (name 'nonguix)
        (url "https://gitlab.com/nonguix/nonguix")
        ;; Enable signature verification:
        (introduction
         (make-channel-introduction
          "897c1a470da759236cc11798f4e0a5f7d4d59fbc"
          (openpgp-fingerprint
           "2A39 3FFF 68F4 EF7A 3D29  12AF 6F51 20A0 22FB B2D5"))))
       %default-channels)"""

science_channel = """
(cons* (channel
  (name 'guix-science)
  (url "https://github.com/guix-science/guix-science.git")
  (introduction
   (make-channel-introduction
        "b1fe5aaff3ab48e798a4cce02f0212bc91f423dc"
        (openpgp-fingerprint
         "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446"))))
         %default-channels)

"""

hpc_channel = """
(cons (channel
        (name 'guix-hpc-non-free)
        (url "https://gitlab.inria.fr/guix-hpc/guix-hpc-non-free.git"))
      %default-channels)
"""

default_guix_packages = [
    "python",
    "bash-minimal",
    "glibc-locales",
    "nss-certs",
    "coreutils",
    "git",
    "make",
    "zlib",
    "python-toolchain",
    "poetry",
    "cuda-toolkit",
    # "cudnn",
]


def __func_to_script(func):
    fun_source = inspect.getsource(func)
    name = fun_source.split("\n")[1]  # skip decorator

    m = re.search(
        "(?<= )\w+", name
    )  # isolate the name of function to use as script name

    source = "\n" + dedent("\n".join(fun_source.split("\n")[2:]))

    filename = os.path.join(temp_dir, m.group(0) + ".py")

    with open(filename, "w") as f:
        f.write(dedent(source))
    return filename


def with_venv(import_libs, python_ver=None):
    def wrap(func):
        assert NOX_INSTALLED
        filename = __func_to_script(func)

        @nox.session(name=m.group(0), python=python_ver)
        def myscript(session):
            for lib in import_libs:
                session.install(lib, silent=True)
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

    temp_dir.cleanup()


def with_guix(
    import_libs,
    with_guix_torch=False,
    with_guix_jax=False,
    manifest=None,
    channels=None,
    python_ver=None,
):
    os.chdir(temp_dir)
    if channels is None:
        channels = os.path.join(temp_dir, "channels.scm")

    if not (os.path.isfile(channels)):
        with open(channels, "w") as ch_file:
            with subprocess.Popen(
                ["guix", "describe", "-f", "channels"], stdout=subprocess.PIPE
            ) as proc:
                channel_str = proc.stdout.read().decode()
                channel_str += (
                    "\n" + hpc_channel
                )  # always add the hpc channel for nvidia drivers
                if with_guix_torch:
                    channel_str += "\n" + nonguix_channel
                if with_guix_jax:
                    channel_str += "\n" + science_channel
                ch_file.write(channel_str)
    if manifest is None:
        packages = default_guix_packages
        if with_guix_torch:
            packages.append("python-pytorch")
        if with_guix_jax:
            packages.append("python-jax")
        manifest = os.path.join(temp_dir, "manifest.scm")
        with open(manifest, "w") as mfile:
            mfile.write(
                "(specifications->manifest '(\n\"" + '"\n "'.join(packages) + '"))'
            )

    assert os.path.isfile(manifest), "manifest file not found"

    script = template_script_guix
    script += "export POETRY_CACHE_DIR=" + os.path.join(temp_dir, "poetry_cache")
    poetry_lock = os.path.join(temp_dir, "poetry.lock")
    if not (os.path.isfile(poetry_lock)):
        script += "\npoetry init -n --name my-experiment 1> /dev/null"
        script += "\npoetry config virtualenvs.create false --local"
        script += "\npoetry add " + " ".join(import_libs)
    script += "\npoetry install --no-interaction"

    def wrap(func):
        filename = __func_to_script(func)
        basename = os.path.splitext(filename)[0]

        with open(os.path.join(temp_dir, basename + "_xp.sh"), "w") as fscript:
            fscript.write(script + "\n python " + filename)

        def myscript():
            pass

        return myscript

    return wrap


def run_guix_xp(env_dir=None, verbose=False, keep_build_dir=False):
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    run_file = module.__file__

    if env_dir is None:
        env_dir = os.path.join(os.path.dirname(run_file), "rlberry_env")

    if not (os.path.isdir(env_dir)):
        os.mkdir(env_dir)

    # copy environment into temp directory
    if len(os.listdir(env_dir)) > 0:
        for filename in glob.glob(env_dir + "/*"):
            subprocess.run(["cp", "-r", filename, temp_dir], check=True)

    os.chdir(temp_dir)
    for filename in glob.glob(temp_dir + "/*_xp.sh"):
        subprocess.run(["chmod", "+x", filename], check=True)
        subprocess.run(filename, check=True)

    for filename in glob.glob(temp_dir + "/*"):
        subprocess.run(["cp", "-r", filename, env_dir], check=True)
    if keep_build_dir:
        print("The build directory has been kept and is located at " + temp_dir)
    else:
        shutil.rmtree(dirpath)
