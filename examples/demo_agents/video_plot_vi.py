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

from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.envs.finite import Chain

env = Chain()
agent = ValueIterationAgent(env, gamma=0.95)
info = agent.fit()
print(info)

env.enable_rendering()
state, info = env.reset()
for tt in range(50):
    action = agent.policy(state)
    next_s, _, done, _ = env.step(action)
    if done:
        break
    state = next_s
video = env.save_video("_video/video_plot_vi.mp4")
