""" 
 ===================== 
 Demo: demo_discretize_state 
 =====================
"""
from rlberry.envs.classic_control import MountainCar
from rlberry.wrappers import DiscretizeStateWrapper

cont_env = MountainCar()
env = DiscretizeStateWrapper(cont_env, 10)  # 10 bins per dimension

print(cont_env.observation_space)
print(env.observation_space)
print("reset in discrete environment gives initial state = ", env.reset()[0])

env.enable_rendering()
for tt in range(20):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    env.sample(54, 1)
env.render()
