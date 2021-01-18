from rlberry.experiment import experiment_generator
from rlberry.agents.kernel_based.rs_ucbvi import RSUCBVIAgent

import numpy as np
import rlberry.seeding as sd


def test_mock_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ['', 'rlberry/experiment/tests/params_experiment.yaml']
        )
    random_numbers = []

    for agent_stats in experiment_generator():
        rng = sd.get_rng()
        random_numbers.append(rng.uniform(size=10))

        assert agent_stats.agent_class is RSUCBVIAgent
        assert agent_stats.init_kwargs['n_episodes'] == 100
        assert agent_stats.init_kwargs['horizon'] == 50

        assert agent_stats.init_kwargs['lp_metric'] == 2
        assert agent_stats.init_kwargs['min_dist'] == 0.0
        assert agent_stats.init_kwargs['max_repr'] == 800
        assert agent_stats.init_kwargs['bonus_scale_factor'] == 1.0
        assert agent_stats.init_kwargs['reward_free'] is True

        assert agent_stats.eval_horizon == 51

        assert agent_stats.train_env.reward_free is False
        assert agent_stats.train_env.array_observation is True

        if agent_stats.agent_name == 'rsucbvi':
            assert agent_stats.init_kwargs['gamma'] == 1.0

        elif agent_stats.agent_name == 'rsucbvi_alternative':
            assert agent_stats.init_kwargs['gamma'] == 0.9

        else:
            raise ValueError()

    #  check that seeding is the same for each AgentStats instance
    for ii in range(1, len(random_numbers)):
        assert np.array_equal(random_numbers[ii-1], random_numbers[ii])
