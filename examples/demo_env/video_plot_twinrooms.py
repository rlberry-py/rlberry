"""
===============================
A demo of twinrooms environment
===============================
 Illustration of TwinRooms environment

.. video:: ../../video_plot_twinrooms.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_twinrooms.jpg'

from rlberry-research.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry-scool.agents.mbqvi import MBQVIAgent
from rlberry.wrappers.discretize_state import DiscretizeStateWrapper
from rlberry.seeding import Seeder

seeder = Seeder(123)

env = TwinRooms()
env = DiscretizeStateWrapper(env, n_bins=20)
env.reseed(seeder)
horizon = 20
agent = MBQVIAgent(env, n_samples=10, gamma=1.0, horizon=horizon)
agent.reseed(seeder)
agent.fit()

observation, info = env.reset()
env.enable_rendering()
for ii in range(10):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if (ii + 1) % horizon == 0:
        observation, info = env.reset()

env.render()
video = env.save_video("_video/video_plot_twinrooms.mp4")
