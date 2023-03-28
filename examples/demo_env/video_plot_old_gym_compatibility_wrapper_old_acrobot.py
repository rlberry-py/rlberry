"""
===============================================
A demo of OldGymCompatibilityWrapper with old_Acrobot environment
===============================================
Illustration of the wrapper for old environments (old Acrobot).

.. video:: ../../video_plot_old_gym_acrobot.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_old_gym_acrobot.jpg'


from rlberry.wrappers.tests.old_env.old_acrobot import Old_Acrobot
from rlberry.agents import RSUCBVIAgent
from rlberry.wrappers import RescaleRewardWrapper
from rlberry.wrappers.gym_utils import OldGymCompatibilityWrapper

env = Old_Acrobot()
env = OldGymCompatibilityWrapper(env)
env = RescaleRewardWrapper(env, (0.0, 1.0))
n_episodes = 300
agent = RSUCBVIAgent(
    env, gamma=0.99, horizon=300, bonus_scale_factor=0.01, min_dist=0.25
)
result = env.reset(seed=42)

agent.fit(budget=n_episodes)

env.enable_rendering()
observation, info = env.reset()
for tt in range(2 * agent.horizon):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Save video
video = env.save_video("_video/video_plot_old_gym_acrobot.mp4")
