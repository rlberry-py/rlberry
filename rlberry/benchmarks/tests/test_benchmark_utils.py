import os
import shutil
from rlberry.benchmarks.benchmark_utils import download_benchmark_from_SB3_zoo
import pytest


@pytest.mark.parametrize("agent_class", ["dqn"])
@pytest.mark.parametrize("env", ["PongNoFrameskip-v4_1"])
def test_experiment_manager_and_multiple_managers_seeding(agent_class, env):
    # remove previous dl
    test_folder_path = "./tests_dl"
    if os.path.exists(test_folder_path):
        shutil.rmtree(test_folder_path)
    os.makedirs(test_folder_path)

    download_benchmark_from_SB3_zoo(agent_class, env, download_path=test_folder_path)

    assert os.path.exists(os.path.join(test_folder_path, agent_class, env))

    if os.path.exists(test_folder_path):
        shutil.rmtree(test_folder_path)
