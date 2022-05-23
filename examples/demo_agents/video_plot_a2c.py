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

from rlberry.agents.torch import A2CAgent
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from gym.wrappers import TimeLimit


env = PBall2D()
env = TimeLimit(env, max_episode_steps=256)
n_timesteps = 50_000
agent = A2CAgent(env, gamma=0.99, learning_rate=0.001)
agent.fit(budget=n_timesteps)

env.enable_rendering()

state = env.reset()
for tt in range(200):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

video = env.save_video("_video/video_plot_a2c.mp4")
