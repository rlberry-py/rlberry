"""
===============================
A demo of MountainCar environment
===============================
 Illustration of MountainCar environment

.. video:: ../../video_plot_montain_car.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_montain_car.jpg'

from rlberry.agents.mbqvi import MBQVIAgent
from rlberry.envs.classic_control import MountainCar
from rlberry.wrappers import DiscretizeStateWrapper

_env = MountainCar()
env = DiscretizeStateWrapper(_env, 20)
agent = MBQVIAgent(env, n_samples=40, gamma=0.99)
agent.fit()

env.enable_rendering()
observation,info = env.reset()
for tt in range(200):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

video = env.save_video("_video/video_plot_montain_car.mp4")
