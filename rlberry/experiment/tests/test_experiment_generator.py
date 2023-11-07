from rlberry.experiment import experiment_generator
from rlberry_research.agents.kernel_based.rs_ucbvi import RSUCBVIAgent

import numpy as np


def test_mock_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["", "rlberry/experiment/tests/params_experiment.yaml"]
    )
    random_numbers = []

    for experiment_manager in experiment_generator():
        rng = experiment_manager.seeder.rng
        random_numbers.append(rng.uniform(size=10))

        assert experiment_manager.agent_class is RSUCBVIAgent
        assert experiment_manager._base_init_kwargs["horizon"] == 2
        assert experiment_manager.fit_budget == 3
        assert experiment_manager.eval_kwargs["eval_horizon"] == 4

        assert experiment_manager._base_init_kwargs["lp_metric"] == 2
        assert experiment_manager._base_init_kwargs["min_dist"] == 0.0
        assert experiment_manager._base_init_kwargs["max_repr"] == 800
        assert experiment_manager._base_init_kwargs["bonus_scale_factor"] == 1.0
        assert experiment_manager._base_init_kwargs["reward_free"] is True

        train_env = experiment_manager.train_env[0](**experiment_manager.train_env[1])
        assert train_env.reward_free is False
        assert train_env.array_observation is True

        if experiment_manager.agent_name == "rsucbvi":
            assert experiment_manager._base_init_kwargs["gamma"] == 1.0

        elif experiment_manager.agent_name == "rsucbvi_alternative":
            assert experiment_manager._base_init_kwargs["gamma"] == 0.9

        else:
            raise ValueError()

    #  check that seeding is the same for each ExperimentManager instance
    for ii in range(1, len(random_numbers)):
        assert np.array_equal(random_numbers[ii - 1], random_numbers[ii])
