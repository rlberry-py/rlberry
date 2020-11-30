from rlberry.envs import Acrobot
from rlberry.agents import RSUCBVIAgent
from rlberry.utils.logging import configure_logging
from rlberry.wrappers import RescaleRewardWrapper

configure_logging("DEBUG")

env = Acrobot()
# rescale rewards to [0, 1]
env = RescaleRewardWrapper(env, (0.0, 1.0))

agent = RSUCBVIAgent(env, n_episodes=500, gamma=0.99, horizon=300,
                     bonus_scale_factor=0.01, min_dist=0.25)
agent.fit()

env.enable_rendering()
state = env.reset()
for tt in range(4*agent.horizon):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

env.save_video("acrobot.mp4")
