import gym

from rlberry.agents import RSUCBVIAgent
from rlberry.wrappers import Wrapper
from rlberry.wrappers import RescaleRewardWrapper

gym_env = gym.make('Acrobot-v1')
gym_env.reward_range = (-1.0, 0.0)  # missing in gym implementation
# from gym to rlberry
env = Wrapper(gym_env)

# rescake rewards to [0, 1]
env = RescaleRewardWrapper(env, (0.0, 1.0))

agent = RSUCBVIAgent(env, n_episodes=10, gamma=0.99, horizon=200,
                     bonus_scale_factor=0.1, min_dist=0.2, verbose=4)
agent.fit()

state = env.reset()
for tt in range(200):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
env.close()
