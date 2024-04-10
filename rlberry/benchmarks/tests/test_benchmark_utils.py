import os
import shutil
from rlberry.benchmarks.benchmark_utils import download_benchmark_from_SB3_zoo
import pytest


@pytest.mark.parametrize("agent_class", ["dqn"])
@pytest.mark.parametrize("env", ["PongNoFrameskip-v4_1"])
def test_experiment_manager_and_multiple_managers_seeding(agent_class, env):
    # remove previous test if existing
    test_folder_path = "./tests_dl"
    if os.path.exists(test_folder_path):
        shutil.rmtree(test_folder_path)
    os.makedirs(test_folder_path)

    # download benchmark
    ret_value = download_benchmark_from_SB3_zoo(
        agent_class, env, download_path=test_folder_path
    )

    # tests expected result
    environment_base_name = env.split("_")[0]
    assert test_folder_path + "/" + agent_class + "/" + env == ret_value
    assert os.path.exists(os.path.join(test_folder_path, agent_class, env))
    assert os.path.exists(
        os.path.join(test_folder_path, agent_class, env, "0.monitor.csv")
    )
    assert os.path.exists(
        os.path.join(test_folder_path, agent_class, env, environment_base_name + ".zip")
    )
    assert os.path.exists(
        os.path.join(test_folder_path, agent_class, env, "evaluations.npz")
    )
    assert os.path.exists(
        os.path.join(
            test_folder_path, agent_class, env, environment_base_name, "args.yml"
        )
    )
    assert os.path.exists(
        os.path.join(
            test_folder_path, agent_class, env, environment_base_name, "config.yml"
        )
    )
    assert os.path.exists(
        os.path.join(
            test_folder_path,
            agent_class,
            env,
            environment_base_name,
            "vecnormalize.pkl",
        )
    )

    if os.path.exists(test_folder_path):
        shutil.rmtree(test_folder_path)
