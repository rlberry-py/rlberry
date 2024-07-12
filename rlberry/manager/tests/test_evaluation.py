import pytest
from rlberry.agents import AgentWithSimplePolicy
from rlberry.manager import evaluation
from rlberry.manager import ExperimentManager
from rlberry_scool.envs import Chain
from rlberry.wrappers import WriterWrapper
import tempfile
import numpy as np
import time


class RandomAgent(AgentWithSimplePolicy):
    name = "RandomAgent"

    def __init__(self, env, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.env = WriterWrapper(self.env, self.writer, write_scalar="reward")

    def fit(self, budget=100, **kwargs):
        observation, info = self.env.reset()
        for ep in range(
            budget + np.random.randint(5)
        ):  # to simulate having different sizes
            action = self.policy(observation)
            observation, reward, done, _, _ = self.env.step(action)

    def policy(self, observation):
        return self.env.action_space.sample()  # choose an action at random


class RandomAgent2(RandomAgent):
    name = "RandomAgent2"


def _create_and_fit_experiment_manager(
    output_dir, outdir_id_style, agent_class=RandomAgent
):
    env_ctor = Chain
    env_kwargs = dict(L=3, fail_prob=0.5)

    manager = ExperimentManager(
        agent_class,
        (env_ctor, env_kwargs),
        fit_budget=15,
        n_fit=3,
        output_dir=output_dir,
        outdir_id_style=outdir_id_style,
        seed=42,
    )
    manager.fit()
    saved_path = manager.save()

    return saved_path


def _create_and_fit_5_experiment_manager(
    output_dir, outdir_id_style, agent_class=RandomAgent
):
    list_path = []

    for i in range(0, 5):
        saved_path = _create_and_fit_experiment_manager(
            output_dir, outdir_id_style, agent_class
        )
        list_path.append(saved_path)
        time.sleep(2)
    return list_path


@pytest.mark.parametrize("outdir_id_style", [None, "unique", "timestamp"])
def test_get_latest_pickle_manager_obj(outdir_id_style):
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = tmpdirname + "/rlberry_data"
        path_list = _create_and_fit_5_experiment_manager(output_dir, outdir_id_style)

        expected_result = path_list[-1]  # latest added, is the more recent
        function_result = evaluation._get_latest_pickle_manager_obj(output_dir)

        assert expected_result == function_result


@pytest.mark.parametrize("outdir_id_style", [None, "unique", "timestamp"])
def test_get_paths_multi_latest_pickle_manager_obj(outdir_id_style):
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = tmpdirname + "/rlberry_data"
        path_list = _create_and_fit_5_experiment_manager(
            output_dir, outdir_id_style, RandomAgent
        )
        expected_result1 = path_list[-1]  # latest added, is the more recent
        path_list = _create_and_fit_5_experiment_manager(
            output_dir, outdir_id_style, RandomAgent2
        )
        expected_result2 = path_list[-1]  # latest added, is the more recent

        function_result = evaluation._get_paths_multi_latest_pickle_manager_obj(
            output_dir
        )

        assert len(function_result) == 2
        assert expected_result1 in function_result
        assert expected_result2 in function_result


@pytest.mark.parametrize("many_agent_by_str_datasource", [True, False])
def test_read_writer_data_single_experiment(many_agent_by_str_datasource):
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = tmpdirname + "/rlberry_data"

        env_ctor = Chain
        env_kwargs = dict(L=3, fail_prob=0.5)

        manager = ExperimentManager(
            RandomAgent,
            (env_ctor, env_kwargs),
            fit_budget=15,
            n_fit=3,
            output_dir=output_dir,
            seed=42,
        )
        manager.fit()

        training_info_experiment = evaluation.read_writer_data(manager)
        training_info_str = evaluation.read_writer_data(
            output_dir, many_agent_by_str_datasource
        )

        assert training_info_experiment.equals(training_info_str)
        assert list(training_info_experiment.columns) == [
            "name",
            "tag",
            "value",
            "dw_time_elapsed",
            "global_step",
            "n_simu",
        ]


@pytest.mark.parametrize("many_agent_by_str_datasource", [True, False])
def test_read_writer_data_list_experiment(many_agent_by_str_datasource):
    env_ctor = Chain
    env_kwargs = dict(L=3, fail_prob=0.5)

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir1 = tmpdirname + "/rlberry_data/agent1"
        manager1 = ExperimentManager(
            RandomAgent,
            (env_ctor, env_kwargs),
            fit_budget=15,
            n_fit=3,
            output_dir=output_dir1,
            seed=42,
        )
        manager1.fit()

        output_dir2 = tmpdirname + "/rlberry_data/agent2"
        manager2 = ExperimentManager(
            RandomAgent2,
            (env_ctor, env_kwargs),
            fit_budget=15,
            n_fit=3,
            output_dir=output_dir2,
            seed=42,
        )
        manager2.fit()

        list_experiment = [manager1, manager2]
        list_str = [output_dir1, output_dir2]

        training_info_experiment = evaluation.read_writer_data(list_experiment)
        training_info_str = evaluation.read_writer_data(
            list_str, many_agent_by_str_datasource
        )

        assert training_info_experiment.equals(training_info_str)
        assert list(training_info_experiment.columns) == [
            "name",
            "tag",
            "value",
            "dw_time_elapsed",
            "global_step",
            "n_simu",
        ]
