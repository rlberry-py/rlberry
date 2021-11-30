"""
===============================================
A demo of Acrobot environment with RSUCBVIAgent
===============================================
"""

from rlberry.envs import Acrobot
from rlberry.agents import RSUCBVIAgent
from rlberry.wrappers import RescaleRewardWrapper

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

env = Acrobot()
# rescale rewards to [0, 1]
env = RescaleRewardWrapper(env, (0.0, 1.0))
n_episodes = 300
agent = RSUCBVIAgent(
    env,
    gamma=0.99,
    horizon=300,
    bonus_scale_factor=0.01, min_dist=0.25)
agent.fit(budget=n_episodes)

env.enable_rendering()
state = env.reset()
for tt in range(4 * agent.horizon):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

# Play the video
fig, ax = plt.subplots()
video = env.get_video()

img = plt.imshow(video[0])
def animate(i):
    img.set_data(video[i])
    return (img,)

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=np.arange(len(video)),
                               interval=10, blit=True)
plt.show()
