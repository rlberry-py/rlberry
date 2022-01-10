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
agent = RSUCBVIAgent(env, gamma=0.99, horizon=horizon,
                     bonus_scale_factor=0.1)
agent.fit(budget=500)

env.enable_rendering()
state = env.reset()
for tt in range(200):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

video = env.save_video("_video/video_plot_rsucbvi.mp4")
