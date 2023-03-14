"""
==================================================
A demo of MBQVI algorithm in Gridworld environment
==================================================
 Illustration of how to set up an MBQVI algorithm in rlberry.
 The environment chosen here is GridWorld environment.

.. video:: ../../video_plot_mbqvi.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_mbqvi.jpg'
from rlberry.agents.mbqvi import MBQVIAgent
from rlberry.envs.finite import GridWorld

params = {}
params["n_samples"] = 100  # samples per state-action pair
params["gamma"] = 0.99
params["horizon"] = None

env = GridWorld(7, 10, walls=((2, 2), (3, 3)), success_probability=0.6)
agent = MBQVIAgent(env, **params)
info = agent.fit()
print(info)

# evaluate policy in a deterministic version of the environment
env_eval = GridWorld(7, 10, walls=((2, 2), (3, 3)), success_probability=1.0)
env_eval.enable_rendering()
state, info = env_eval.reset()
for tt in range(50):
    action = agent.policy(state)
    next_s, _, _, _ = env_eval.step(action)
    state = next_s
video = env_eval.save_video("_video/video_plot_mbqvi.mp4")
