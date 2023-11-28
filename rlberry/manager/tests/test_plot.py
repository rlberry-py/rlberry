import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
import sys

from rlberry.wrappers import WriterWrapper
from rlberry_research.envs import GridWorld
from rlberry.manager import plot_writer_data, ExperimentManager, read_writer_data
from rlberry_scool.agents import UCBVIAgent


class VIAgent(UCBVIAgent):
    name = "UCBVIAgent"

    def __init__(self, env, **kwargs):
        UCBVIAgent.__init__(self, env, horizon=50, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")


def _create_and_fit_experiment_manager(output_dir, outdir_id_style):
    env_ctor = GridWorld
    env_kwargs = dict(nrows=2, ncols=2, reward_at={(1, 1): 0.1, (2, 2): 1.0})

    manager = ExperimentManager(
        VIAgent,
        (env_ctor, env_kwargs),
        fit_budget=10,
        n_fit=3,
        output_dir=output_dir,
        outdir_id_style=outdir_id_style,
    )
    manager.fit()
    manager.save()
    return manager


def _compute_reward(rewards):
    return np.cumsum(rewards)


@pytest.mark.parametrize("outdir_id_style", [None, "unique", "timestamp"])
def test_plot_writer_data_with_manager_input(outdir_id_style):
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = tmpdirname + "/rlberry_data"
        manager = _create_and_fit_experiment_manager(output_dir, outdir_id_style)
        os.system("ls " + tmpdirname + "/rlberry_data/manager_data")

        # Plot of the cumulative reward
        data_source = manager
        output = plot_writer_data(
            data_source,
            tag="reward",
            preprocess_func=_compute_reward,
            title="Cumulative Reward",
            show=False,
            savefig_fname=tmpdirname + "/test.png",
        )
        assert (
            os.path.getsize(tmpdirname + "/test.png") > 1000
        ), "plot_writer_data saved an empty image"
        assert len(output) > 1


def test_ci():
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = tmpdirname + "/rlberry_data"
        manager = _create_and_fit_experiment_manager(output_dir, None)
        os.system("ls " + tmpdirname + "/rlberry_data/manager_data")

        # Plot of the cumulative reward
        data_source = manager
        output = plot_writer_data(
            data_source,
            tag="reward",
            preprocess_func=_compute_reward,
            title="Cumulative Reward",
            show=False,
            savefig_fname=tmpdirname + "/test.png",
        )
        assert (
            os.path.getsize(tmpdirname + "/test.png") > 1000
        ), "plot_writer_data saved an empty image"
        assert len(output) > 1


@pytest.mark.xfail(sys.platform == "win32", reason="bug with windows???")
@pytest.mark.parametrize("outdir_id_style", ["timestamp"])
def test_plot_writer_data_with_directory_input(outdir_id_style):
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = tmpdirname + "/rlberry_data"
        manager = _create_and_fit_experiment_manager(output_dir, outdir_id_style)
        del manager

        os.system("ls " + tmpdirname + "/rlberry_data/manager_data")

        #
        # Single directory
        #

        data_source = output_dir
        output = plot_writer_data(
            data_source,
            tag="reward",
            preprocess_func=_compute_reward,
            title="Cumulative Reward",
            show=False,
            savefig_fname=tmpdirname + "/test.png",
        )
        assert (
            os.path.getsize(tmpdirname + "/test.png") > 1000
        ), "plot_writer_data saved an empty image"
        assert len(output) > 1

        list_dirs = list(Path(tmpdirname + "/rlberry_data/manager_data").iterdir())
        list_dirs = [str(dir) for dir in list_dirs]

        #
        # List of directories
        #
        output_with_list_dirs = plot_writer_data(
            list_dirs,
            tag="reward",
            preprocess_func=_compute_reward,
            title="Cumulative Reward",
            show=False,
            savefig_fname=tmpdirname + "/test.png",
        )

        assert np.all(output.shape == output_with_list_dirs.shape)

        output = plot_writer_data(
            data_source,
            tag="reward",
            xtag="dw_time_elapsed",
            smooth=True,
            preprocess_func=_compute_reward,
            title="Cumulative Reward",
            show=False,
            savefig_fname=tmpdirname + "/test.png",
        )
        output = read_writer_data(data_source)
