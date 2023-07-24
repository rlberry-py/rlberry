from rlberry.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry.agents.mbqvi import MBQVIAgent
from rlberry.manager import ExperimentManager, evaluate_agents
import numpy as np

params = {}
params["n_samples"] = 8  # samples per state-action pair
params["gamma"] = 0.9
params["horizon"] = None


# hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
def test_mbqvi_applegold():
    rbagent = ExperimentManager(
        MBQVIAgent,
        (AppleGold, None),
        init_kwargs=params,
        fit_budget=1000,
        n_fit=16,
        parallelization="process",
        mp_context="fork",
        seed=42,
        eval_kwargs=dict(eval_horizon=1000),
    )

    rbagent.fit()
    evaluation = evaluate_agents([rbagent], n_simulations=16, show=False).values
    assert np.mean(evaluation) > 470
