"""
=============================================================
A demo of RSKernelUCBVIAgent algorithm in Acrobot environment
=============================================================
 Illustration of how to set up a RSKernelUCBVI algorithm in rlberry.
 The environment chosen here is Acrobot environment.

.. video:: ../../video_plot_rs_kernel_ucbvi.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_rs_kernel_ucbvi.jpg'

from rlberry_research.envs import Acrobot
from rlberry_research.agents import RSKernelUCBVIAgent
from rlberry.wrappers import RescaleRewardWrapper

env = Acrobot()
# rescake rewards to [0, 1]
env = RescaleRewardWrapper(env, (0.0, 1.0))

agent = RSKernelUCBVIAgent(
    env,
    gamma=0.99,
    horizon=300,
    bonus_scale_factor=0.01,
    min_dist=0.2,
    bandwidth=0.05,
    beta=1.0,
    kernel_type="gaussian",
)
agent.fit(budget=500)

env.enable_rendering()
observation, info = env.reset()

time_before_done = 0
ended = False
for tt in range(2 * agent.horizon):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if not done and not ended:
        time_before_done += 1
    if done:
        ended = True

print("steps to achieve the goal for the first time = ", time_before_done)
video = env.save_video("_video/video_plot_rs_kernel_ucbvi.mp4")
