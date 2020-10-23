import gym
from rlberry.wrappers import GymWrapper 
from rlberry.agents import RSUCBVIAgent 
from rlberry.wrappers import DiscretizeStateWrapper 

gym_env = gym.make('MountainCar-v0')

# from gym to rlberry
env = GymWrapper(gym_env)

agent = RSUCBVIAgent(env, n_episodes=2000, gamma=0.99, horizon=200, bonus_scale_factor=0.1, verbose=4)
agent.fit()

state = env.reset()
for tt in range(200):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
env.close()