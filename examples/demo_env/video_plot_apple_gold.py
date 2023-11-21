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
from rlberry_research.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry_research.agents.dynprog import ValueIterationAgent

env = AppleGold(reward_free=False, array_observation=False)

agent = ValueIterationAgent(env, gamma=0.9)
info = agent.fit()
print(info)

env.enable_rendering()

observation, info = env.reset()
for tt in range(5):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        break
env.render()
video = env.save_video("_video/video_plot_apple_gold.mp4")
