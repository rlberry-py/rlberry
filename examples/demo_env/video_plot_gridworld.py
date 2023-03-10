"""
.. _gridworld_example:

========================================================
A demo of Gridworld environment with ValueIterationAgent
========================================================
Illustration of the training and video rendering ofValueIteration Agent in
Gridworld environment.

.. video:: ../../video_plot_gridworld.mp4
   :width: 600
"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_gridworld.jpg'

from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.envs.finite import GridWorld


env = GridWorld(7, 10, walls=((2, 2), (3, 3)))

agent = ValueIterationAgent(env, gamma=0.95)
info = agent.fit()
print(info)

env.enable_rendering()
state, info = env.reset()
for tt in range(50):
    action = agent.policy(state)
    next_s, _, done, _ = env.step(action)
    if done:
        # Warning: this will never happen in the present case because there is no terminal state.
        # See the doc of GridWorld for more informations on the default parameters of GridWorld.
        break
    state = next_s

# Save the video
video = env.save_video("_video/video_plot_gridworld.mp4", framerate=10)
