""" 
 ===================== 
 Demo: demo_chain 
 =====================
"""
from rlberry.envs.finite import Chain

env = Chain(10, 0.1)
env.enable_rendering()
for tt in range(100):
    env.step(env.action_space.sample())
env.render()
