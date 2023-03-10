"""
==============================================
A demo of PPO algorithm in PBall2D environment
==============================================
 Illustration of how to set up an PPO algorithm in rlberry.
 The environment chosen here is PBALL2D environment.

.. video:: ../../video_plot_ppo.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_a2c.jpg'

from rlberry.agents.torch import PPOAgent
from rlberry.envs.benchmarks.ball_exploration import PBall2D


env = PBall2D()
n_steps = 3e3

agent = PPOAgent(env)
agent.fit(budget=n_steps)

env.enable_rendering()
state, info = env.reset()
for tt in range(200):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

video = env.save_video("_video/video_plot_ppo.mp4")
