from rlberry.envs import Acrobot
from rlberry.agents import RSKernelUCBVIAgent
from rlberry.wrappers import RescaleRewardWrapper


env = Acrobot()
# rescake rewards to [0, 1]
env = RescaleRewardWrapper(env, (0.0, 1.0))

agent = RSKernelUCBVIAgent(env, n_episodes=500, gamma=0.99, horizon=300,
                           bonus_scale_factor=0.01,
                           min_dist=0.2, bandwidth=0.05, beta=1.0,
                           kernel_type="gaussian")
agent.fit()

env.enable_rendering()
state = env.reset()

time_before_done = 0
ended = False
for tt in range(4*agent.horizon):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    if not done and not ended:
        time_before_done += 1
    if done:
        ended = True
    state = next_state

print("steps to achieve the goal for the first time = ", time_before_done)
env.save_video("kernel_acrobot_gaussian.mp4", framerate=15)
