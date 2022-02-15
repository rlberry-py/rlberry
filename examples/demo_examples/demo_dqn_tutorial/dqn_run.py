"""
 =====================
 Demo: dqn_run
 =====================

Run simple DQN agent and visualize training data
with tensorboard.

While training, run

$ tensorboard --logdir rlberry_data/dqn_example

to check logs. If you run the example several times,
all the runs will appear on tensorboard. Change logdir
to the folder of a specific run if you'd like to
visualize a single run.
"""

from dqn_agent import DQNAgent
from dqn_nets import QNet
from rlberry.manager import AgentManager
from rlberry.envs import gym_make


#
# Define environment
#
env_id = "CartPole-v0"
env = (gym_make, dict(id=env_id))

#
# Define DQN parameters
#
init_kwargs = dict(
    q_net_constructor=QNet,
    gamma=0.99,
    batch_size=256,
    eval_every=500,
    buffer_capacity=30000,
    update_target_every=500,
    epsilon_start=1.0,
    epsilon_min=0.05,
    decrease_epsilon=200,
    learning_rate=0.001,
)

fit_kwargs = dict(fit_budget=20_000)

#
# Create agent manager and run
#
manager = AgentManager(
    DQNAgent,
    train_env=env,
    enable_tensorboard=True,
    init_kwargs=init_kwargs,
    fit_kwargs=fit_kwargs,
    n_fit=2,
    output_dir="rlberry_data/dqn_example",
)
manager.fit()
