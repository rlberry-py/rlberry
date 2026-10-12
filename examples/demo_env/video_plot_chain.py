"""
===============================
A demo of Chain environment
===============================
 Illustration of Chain environment

.. video:: ../../video_plot_chain.mp4
   :width: 600

"""

# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_chain.jpg'


from rlberry_scool.envs.finite import Chain

env = Chain(10, 0.1)
env.enable_rendering()
for tt in range(5):
    env.step(env.action_space.sample())
env.render()
env.save_video("_video/video_plot_chain.mp4")
