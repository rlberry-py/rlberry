""" 
 ===================== 
 Demo: demo_gym_wrapper 
 =====================
"""
from rlberry.envs import gym_make
from rlberry.agents import RSUCBVIAgent
from rlberry.wrappers import RescaleRewardWrapper

env = gym_make('Acrobot-v1')
env.reward_range = (-1.0, 0.0)  # missing in gym implementation

# rescake rewards to [0, 1]
env = RescaleRewardWrapper(env, (0.0, 1.0))

agent = RSUCBVIAgent(env, gamma=0.99, horizon=200,
                     bonus_scale_factor=0.1, min_dist=0.2)
agent.fit(budget=10)

state = env.reset()
for tt in range(200):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
env.close()
