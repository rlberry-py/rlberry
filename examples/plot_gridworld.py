"""
========================================================
A demo of Gridworld environment with ValueIterationAgent
========================================================
"""

from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.envs.finite import GridWorld
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

fig, ax = plt.subplots()

env = GridWorld(7, 10, walls=((2, 2), (3, 3)))
agent = ValueIterationAgent(env, gamma=0.95)
info = agent.fit()
print(info)

env.enable_rendering()

state = env.reset()
for tt in range(50):
    action = agent.policy(state)
    next_s, _, done, _ = env.step(action)
    if done:
        break
    state = next_s
video = env.get_video()

img = plt.imshow(video[0])
def animate(i):
    img.set_data(video[i])
    return (img,)

# call the animator. blit=True means only re-draw the parts that have changed.
# *interval* draws a new frame every *interval* milliseconds.
anim = animation.FuncAnimation(fig, animate, frames=np.arange(len(video)),blit=True)
plt.show()
