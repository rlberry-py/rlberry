"""
===============================
A demo of AppleGold environment
===============================
 Illustration of Applegold environment on which we train a ValueIteration
 algorithm.

.. video:: ../../video_plot_apple_gold.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_apple_gold.jpg'
from rlberry.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry.agents.dynprog import ValueIterationAgent

env = AppleGold(reward_free=False, array_observation=False)

agent = ValueIterationAgent(env, gamma=0.9)
info = agent.fit()
print(info)

env.enable_rendering()

state, info = env.reset()
for tt in range(5):
    action = agent.policy(state)
    next_s, _, done, _ = env.step(action)
    if done:
        break
    state = next_s
env.render()
video = env.save_video("_video/video_plot_apple_gold.mp4")
