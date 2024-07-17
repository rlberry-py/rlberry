import pytest
from rlberry.agents import AgentWithSimplePolicy
from rlberry.utils import loading_tools
from rlberry.manager import ExperimentManager
from rlberry_scool.envs import Chain
import tempfile
import numpy as np
import time


class RandomAgent(AgentWithSimplePolicy):
    name = "RandomAgent"

    def __init__(self, env, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, writer_extra="reward", **kwargs)

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
def test_get_single_path_of_most_recently_trained_experiment_manager_obj_from_path(
    outdir_id_style,
):
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = tmpdirname + "/rlberry_data"
        path_list = _create_and_fit_5_experiment_manager(output_dir, outdir_id_style)

        expected_result = path_list[-1]  # latest added, is the more recent
        function_result = loading_tools.get_single_path_of_most_recently_trained_experiment_manager_obj_from_path(
            output_dir
        )

        assert expected_result == function_result


@pytest.mark.parametrize("outdir_id_style", [None, "unique", "timestamp"])
def test_get_all_path_of_most_recently_trained_experiments_manager_obj_from_path(
    outdir_id_style,
):
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

        function_result = loading_tools.get_all_path_of_most_recently_trained_experiments_manager_obj_from_path(
            output_dir
        )

        assert len(function_result) == 2
        assert expected_result1 in function_result
        assert expected_result2 in function_result
