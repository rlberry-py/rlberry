"""
===============================================
A demo of SpringCartPole environment with DQNAgent
===============================================
Illustration of the training and video rendering of DQN Agent in
SpringCartPole environment.

Agent is not tuned and does not learn. This is just for illustration purpose.

.. video:: ../../video_plot_springcartpole.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_springcartpole.jpg'

from rlberry.envs.classic_control import SpringCartPole
from rlberry.agents.torch import DQNAgent

model_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": (256, 256),
    "reshape": False,
}

init_kwargs = dict(
    q_net_constructor="rlberry.agents.torch.utils.training.model_factory_from_env",
    q_net_kwargs=model_configs,
)

env = SpringCartPole()
agent = DQNAgent(env, **init_kwargs)
agent.fit(budget=1e4)

env.enable_rendering()
state = env.reset()
for tt in range(1000):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

# Save video
video = env.save_video("_video/video_plot_springcartpole.mp4")
