"""
===============================================
A demo of DQN algorithm in CartPole environment
===============================================
Illustration of how to set up a DQN algorithm in rlberry.
The environment chosen here is gym's cartpole environment.

As DQN can be computationally intensive and hard to tune,
one can use tensorboard to visualize the training of the DQN
using the following command:

.. code-block:: bash

    tensorboard --logdir {Path(agent.writer.log_dir).parent}

.. video:: ../../video_plot_dqn.mp4
   :width: 600

"""

# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_dqn.jpg'

from rlberry.envs import gym_make
from torch.utils.tensorboard import SummaryWriter

from rlberry.agents.torch.dqn import DQNAgent
from rlberry.utils.logging import configure_logging

from gym.wrappers.monitoring import video_recorder


configure_logging(level="INFO")

env = gym_make("CartPole-v1")
agent = DQNAgent(env, epsilon_decay_interval=1000)
agent.set_writer(SummaryWriter())

print(f"Running DQN on {env}")

agent.fit(budget=50)
vid = video_recorder.VideoRecorder(
    env,
    path="_video/video_plot_dqn.mp4",
    enabled=True,
)

for episode in range(3):
    done = False
    state = env.reset()
    while not done:
        action = agent.policy(state)
        state, reward, done, _ = env.step(action)
        vid.capture_frame()
env.close()
