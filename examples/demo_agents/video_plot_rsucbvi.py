"""
==================================================
A demo of RSUCBVI algorithm in MountainCar environment
==================================================
 Illustration of how to set up an RSUCBVI algorithm in rlberry.
 The environment chosen here is MountainCar environment.

.. video:: ../../video_plot_rsucbvi.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_rsucbvi.jpg'

from rlberry.agents import RSUCBVIAgent
from rlberry.envs.classic_control import MountainCar

env = MountainCar()
horizon = 170
print("Running RS-UCBVI on %s" % env.name)
agent = RSUCBVIAgent(env, gamma=0.99, horizon=horizon, bonus_scale_factor=0.1)
agent.fit(budget=500)

env.enable_rendering()
observation,info = env.reset()
for tt in range(200):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

video = env.save_video("_video/video_plot_rsucbvi.mp4")
