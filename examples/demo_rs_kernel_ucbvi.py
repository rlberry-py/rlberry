import gym
from rlberry.envs import Acrobot
from rlberry.agents import RSKernelUCBVIAgent
from rlberry.wrappers import RescaleRewardWrapper


env = Acrobot()
# rescake rewards to [0, 1]
env = RescaleRewardWrapper(env, (0.0, 1.0))

agent = RSKernelUCBVIAgent(env, n_episodes=500, gamma=0.99, horizon=300, bonus_scale_factor=0.01,
                           min_dist=0.2, bandwidth=0.05, beta=1.0, kernel_type="gaussian", verbose=4)
agent.fit()

env.enable_rendering()
state = env.reset()
for tt in range(4*agent.horizon):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

env.save_video("kernel_acrobot_gaussian.mp4", framerate=15)
