""" 
 ===================== 
 Demo: demo_mountain_car 
 =====================
"""
from rlberry.agents.mbqvi import MBQVIAgent
from rlberry.envs.classic_control import MountainCar
from rlberry.wrappers import DiscretizeStateWrapper

_env = MountainCar()
env = DiscretizeStateWrapper(_env, 20)
agent = MBQVIAgent(env, n_samples=40, gamma=0.99)
agent.fit()

env.enable_rendering()
state = env.reset()
for tt in range(200):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

env.save_video("mountain_car.mp4", framerate=50)
