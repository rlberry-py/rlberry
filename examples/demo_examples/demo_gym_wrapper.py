""" 
 ===================== 
 Demo: demo_gym_wrapper 
 =====================
"""
from rlberry.envs import gym_make
from rlberry.agents import RSUCBVIAgent
from rlberry.wrappers import RescaleRewardWrapper

env = gym_make("Acrobot-v1")
env.reward_range = (-1.0, 0.0)  # missing in gym implementation

# rescake rewards to [0, 1]
env = RescaleRewardWrapper(env, (0.0, 1.0))

agent = RSUCBVIAgent(env, gamma=0.99, horizon=200, bonus_scale_factor=0.1, min_dist=0.2)
agent.fit(budget=10)

observation,info = env.reset()
for tt in range(200):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info  = env.step(action)
    done = terminated or truncated
    env.render()
env.close()
