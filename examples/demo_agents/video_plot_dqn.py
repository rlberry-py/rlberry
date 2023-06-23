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

from gymnasium.wrappers.record_video import RecordVideo
import shutil
import os


configure_logging(level="INFO")

env = gym_make("CartPole-v1", render_mode="rgb_array")
agent = DQNAgent(env, epsilon_decay_interval=1000)
agent.set_writer(SummaryWriter())

print(f"Running DQN on {env}")

agent.fit(budget=50)
env = RecordVideo(env, "_video/temp")

for episode in range(3):
    done = False
    observation, info = env.reset()
    while not done:
        action = agent.policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
env.close()

# need to move the final result inside the folder used for documentation
os.rename("_video/temp/rl-video-episode-0.mp4", "_video/video_plot_dqn.mp4")
shutil.rmtree("_video/temp/")
