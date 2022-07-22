"""
.. _dqn_example:

=====================
Demo: demo_dqn_lambda
=====================
"""

from rlberry.agents.torch import DQNAgent
from rlberry.envs import gym_make
from rlberry.manager import AgentManager, MultipleManagers
from rlberry.manager.evaluation import evaluate_agents


#
# Define env and different sets of parameters
#
N_FIT = 3
MAX_WORKERS = 3
FIT_KWARGS = dict(fit_budget=100_000)
TRAIN_ENV = (gym_make, {"id": "CartPole-v1"})
ENABLE_TENSORBOARD = True

DQN_PARAMS = dict(
    lambda_=0.0,  # for standard DQN
    chunk_size=1,  # for standard DQN
    batch_size=256,
    eval_interval=500,
)

DQN_LAMBDA_PARAMS = DQN_PARAMS.copy()
DQN_LAMBDA_PARAMS.update(
    dict(
        lambda_=0.5,
        chunk_size=8,
        batch_size=32,
    )
)


#
# Create managers
#
if __name__ == "__main__":
    managers = MultipleManagers(parallelization="thread")

    # Standard DQN
    managers.append(
        AgentManager(
            DQNAgent,
            TRAIN_ENV,
            agent_name="DQN",
            init_kwargs=DQN_PARAMS,
            fit_kwargs=FIT_KWARGS,
            n_fit=N_FIT,
            max_workers=MAX_WORKERS,
            parallelization="process",
            seed=42,
            enable_tensorboard=ENABLE_TENSORBOARD,
            output_dir="temp/dqn_example",
        )
    )

    # DQN with Q(lambda)
    managers.append(
        AgentManager(
            DQNAgent,
            TRAIN_ENV,
            agent_name="DQN + Q($\\lambda$)",
            init_kwargs=DQN_LAMBDA_PARAMS,
            fit_kwargs=FIT_KWARGS,
            n_fit=N_FIT,
            max_workers=MAX_WORKERS,
            parallelization="process",
            seed=42,
            enable_tensorboard=ENABLE_TENSORBOARD,
            output_dir="temp/dqn_example",
        )
    )

    # Fit and plot
    managers.run(save=True)
    evaluate_agents(managers.instances, show=True)
