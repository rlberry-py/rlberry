"""
=======================================================
A demo of ValueIteration algorithm in Chain environment
=======================================================
 Illustration of how to set up an ValueIteration algorithm in rlberry.
 The environment chosen here is Chain environment.

.. video:: ../../video_plot_vi.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_vi.jpg'

from rlberry_scool.agents.dynprog import ValueIterationAgent
from rlberry_research.envs.finite import Chain

env = Chain()
agent = ValueIterationAgent(env, gamma=0.95)
info = agent.fit()
print(info)

env.enable_rendering()
observation, info = env.reset()
for tt in range(50):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        break
video = env.save_video("_video/video_plot_vi.mp4")
