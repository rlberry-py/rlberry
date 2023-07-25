"""
==============================================
A demo of A2C algorithm in PBall2D environment
==============================================
 Illustration of how to set up an A2C algorithm in rlberry.
 The environment chosen here is PBALL2D environment.

.. video:: ../../video_plot_a2c.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_a2c.jpg'

from gymnasium.wrappers import TimeLimit

from rlberry.agents.torch import A2CAgent
from rlberry.envs.benchmarks.ball_exploration import PBall2D

env = PBall2D()
env = TimeLimit(env, max_episode_steps=256)
n_timesteps = 50_000
agent = A2CAgent(env, gamma=0.99, learning_rate=0.001)
agent.fit(budget=n_timesteps)

env.enable_rendering()

observation, info = env.reset()
for tt in range(200):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

video = env.save_video("_video/video_plot_a2c.mp4")
