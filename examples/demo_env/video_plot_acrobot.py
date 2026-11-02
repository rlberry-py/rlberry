"""
===============================================
A demo of Acrobot environment with RSUCBVIAgent
===============================================
Illustration of the training and video rendering of RSUCBVI Agent in Acrobot
environment.

.. video:: ../../video_plot_acrobot.mp4
   :width: 600

"""

# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_acrobot.jpg'

from rlberry_research.envs import Acrobot
from rlberry_research.agents import RSUCBVIAgent
from rlberry.wrappers import RescaleRewardWrapper

env = Acrobot()
# rescale rewards to [0, 1]
env = RescaleRewardWrapper(env, (0.0, 1.0))
n_episodes = 300
agent = RSUCBVIAgent(
    env, gamma=0.99, horizon=300, bonus_scale_factor=0.01, min_dist=0.25
)
agent.fit(budget=n_episodes)

env.enable_rendering()
observation, info = env.reset()
for tt in range(2 * agent.horizon):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Save video
video = env.save_video("_video/video_plot_acrobot.mp4")
