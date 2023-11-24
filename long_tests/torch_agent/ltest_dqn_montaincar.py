from rlberry.envs import gym_make
from rlberry_research.agents.torch import DQNAgent
from rlberry.manager import ExperimentManager, evaluate_agents
import numpy as np

model_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": (256, 256),
    "reshape": False,
}


# hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
def test_dqn_montaincar():
    env_ctor = gym_make
    env_kwargs = dict(id="MountainCar-v0")

    rbagent = ExperimentManager(
        DQNAgent,
        (env_ctor, env_kwargs),
        init_kwargs=dict(
            q_net_constructor="rlberry.agents.torch.utils.training.model_factory_from_env",
            q_net_kwargs=model_configs,
            batch_size=128,
            max_replay_size=10000,
            learning_rate=4e-3,
            learning_starts=1000,
            gamma=0.98,
            train_interval=16,
            gradient_steps=8,
            epsilon_init=0.2,
            epsilon_final=0.07,
            epsilon_decay_interval=600,
        ),
        fit_budget=1.2e5,
        eval_kwargs=dict(eval_horizon=500),
        n_fit=1,
        parallelization="process",
        mp_context="fork",
        seed=42,
    )

    rbagent.fit()
    evaluation = evaluate_agents([rbagent], n_simulations=16, show=False).values
    assert np.mean(evaluation) > -110
