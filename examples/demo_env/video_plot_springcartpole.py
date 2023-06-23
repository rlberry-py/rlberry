"""
===============================================
A demo of SpringCartPole environment with DQNAgent
===============================================
Illustration of the training and video rendering of DQN Agent in
SpringCartPole environment.

Agent is slightly tuned, but not optimal. This is just for illustration purpose.

.. video:: ../../video_plot_springcartpole.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_springcartpole.jpg'

from rlberry.envs.classic_control import SpringCartPole
from rlberry.agents.torch import DQNAgent
from gymnasium.wrappers.time_limit import TimeLimit

model_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": (256, 256),
    "reshape": False,
}

init_kwargs = dict(
    q_net_constructor="rlberry.agents.torch.utils.training.model_factory_from_env",
    q_net_kwargs=model_configs,
)

env = SpringCartPole(obs_trans=False, swing_up=True)
env = TimeLimit(env, max_episode_steps=500)
agent = DQNAgent(env, **init_kwargs)
agent.fit(budget=1e5)

env.enable_rendering()
observation, info = env.reset()

for tt in range(1000):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        observation, info = env.reset()

# Save video
video = env.save_video("_video/video_plot_springcartpole.mp4")
