"""
========================================================
A demo of Gridworld environment with ValueIterationAgent
========================================================
Illustration of the training and video rendering ofValueIteration Agent in
Gridworld environment.

.. video:: ../video_plot_gridworld.mp4
   :width: 600
"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_gridworld.jpg'

from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.envs.finite import GridWorld


fig, ax = plt.subplots()

env = GridWorld(7, 10, walls=((2, 2), (3, 3)))
agent = ValueIterationAgent(env, gamma=0.95)
info = agent.fit()
print(info)

env.enable_rendering()

state = env.reset()
for tt in range(50):
    action = agent.policy(state)
    next_s, _, done, _ = env.step(action)
    if done:
        break
    state = next_s

# Save the video
video = env.save_video("../docs/_video/video_plot_gridworld.mp4", framerate=10)
