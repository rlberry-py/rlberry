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

from rlberry.envs import Acrobot
from rlberry.agents import RSUCBVIAgent
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
state = env.reset()
for tt in range(2 * agent.horizon):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

# Save video
video = env.save_video("_video/video_plot_acrobot.mp4")
