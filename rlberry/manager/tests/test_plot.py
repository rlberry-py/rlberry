import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
import pandas as pd
import sys
import matplotlib.pyplot as plt

from rlberry.wrappers import WriterWrapper
from rlberry_research.envs import Chain
from rlberry.manager import plot_writer_data, ExperimentManager, read_writer_data
from rlberry.manager.plotting import plot_smoothed_curves, plot_synchronized_curves
from rlberry.agents import AgentWithSimplePolicy


class RandomAgent(AgentWithSimplePolicy):
    name = "RandomAgent"

    def __init__(self, env, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")

    def fit(self, budget=100, **kwargs):
        observation, info = self.env.reset()
        for ep in range(budget):
            action = self.policy(observation)
            observation, reward, done, _, _ = self.env.step(action)

    def policy(self, observation):
        return self.env.action_space.sample()  # choose an action at random


def _create_and_fit_experiment_manager(output_dir, outdir_id_style):
    env_ctor = Chain
    env_kwargs = dict(L=3, fail_prob=0.5)

    manager = ExperimentManager(
        RandomAgent,
        (env_ctor, env_kwargs),
        fit_budget=15,
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


@pytest.mark.parametrize("error_representation", ["ci", "pi", "cb", "raw_curves"])
def test_smooth_ci(error_representation):
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = tmpdirname + "/rlberry_data"
        manager = _create_and_fit_experiment_manager(output_dir, None)
        os.system("ls " + tmpdirname + "/rlberry_data/manager_data")

        # Plot of the cumulative reward
        data_source = manager
        output = plot_writer_data(
            data_source,
            tag="reward",
            smooth=True,
            error_representation=error_representation,
            preprocess_func=_compute_reward,
            title="Cumulative Reward",
            show=False,
            linestyles=True,
            savefig_fname=tmpdirname + "/test.png",
        )
        assert (
            os.path.getsize(tmpdirname + "/test.png") > 1000
        ), "plot_writer_data saved an empty image"
        assert len(output) > 1

        # Plot of the cumulative reward
        data_source = manager
        output = plot_writer_data(
            data_source,
            smoothing_bandwidth=5,
            tag="reward",
            smooth=True,
            error_representation=error_representation,
            preprocess_func=_compute_reward,
            title="Cumulative Reward",
            show=False,
            linestyles=True,
            savefig_fname=tmpdirname + "/test2.png",
        )
        assert (
            os.path.getsize(tmpdirname + "/test2.png") > 1000
        ), "plot_writer_data saved an empty image"
        assert len(output) > 1


@pytest.mark.parametrize("error_representation", ["ci", "pi", "raw_curves"])
def test_non_smooth_ci(error_representation):
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = tmpdirname + "/rlberry_data"
        manager = _create_and_fit_experiment_manager(output_dir, None)
        os.system("ls " + tmpdirname + "/rlberry_data/manager_data")

        # Plot of the cumulative reward
        data_source = manager
        output = plot_writer_data(
            data_source,
            tag="reward",
            smooth=False,
            error_representation=error_representation,
            preprocess_func=_compute_reward,
            title="Cumulative Reward",
            show=False,
            linestyles=True,
            savefig_fname=tmpdirname + "/test.png",
        )
        assert (
            os.path.getsize(tmpdirname + "/test.png") > 1000
        ), "plot_writer_data saved an empty image"
        assert len(output) > 1


def test_without_rlberry():
    df = pd.DataFrame(
        {"name": ["a", "a", "a"], "x": [1, 2, 3], "y": [3, 4, 5], "n_simu": [0, 0, 0]}
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        with plt.ion():  # do not block on plt.show
            plot_smoothed_curves(df, "x", "y", savefig_fname=tmpdirname + "/test.png")
            plot_synchronized_curves(
                df, "x", "y", savefig_fname=tmpdirname + "/test.png"
            )


def test_warning_error_rep():
    msg = "error_representation not implemented"
    with tempfile.TemporaryDirectory() as tmpdirname:
        with pytest.raises(ValueError, match=msg):
            output_dir = tmpdirname + "/rlberry_data"
            manager = _create_and_fit_experiment_manager(output_dir, None)
            data_source = manager
            output = plot_writer_data(
                data_source,
                tag="reward",
                smooth=True,
                error_representation="does_not_exist",
                preprocess_func=_compute_reward,
                title="Cumulative Reward",
                show=False,
                linestyles=True,
                savefig_fname=tmpdirname + "/test.png",
            )


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
