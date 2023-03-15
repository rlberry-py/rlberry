"""
 =====================
 Demo: demo_save_video
 =====================
"""
from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.envs.finite import GridWorld

env = GridWorld(7, 10, walls=((2, 2), (3, 3)))
agent = ValueIterationAgent(env, gamma=0.95)
info = agent.fit()
print(info)

env.enable_rendering()

observation, info = env.reset()
for tt in range(200):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        break

env.save_video("gridworld.mp4", framerate=15)
