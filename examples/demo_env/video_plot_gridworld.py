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

from rlberry_scool.agents.dynprog import ValueIterationAgent
from rlberry_scool.envs.finite import GridWorld


env = GridWorld(7, 10, walls=((2, 2), (3, 3)))

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
        # Warning: this will never happen in the present case because there is no terminal state.
        # See the doc of GridWorld for more informations on the default parameters of GridWorld.
        break
# Save the video
env.save_video("_video/video_plot_gridworld.mp4", framerate=10)
