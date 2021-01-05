from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.envs.classic_control import MountainCar
from rlberry.wrappers.vis2d import Vis2dWrapper
from rlberry.agents import RSUCBVIAgent


# env = NRoom(nrooms=10, array_observation=True, reward_free=True)
# env = Vis2dWrapper(env, n_bins_obs=20, memory_size=100)

# agent = RSUCBVIAgent(env, n_episodes=500, gamma=0.99, horizon=200,
#                      bonus_scale_factor=0.1, copy_env=False, min_dist=0.0)

env = MountainCar()
env = Vis2dWrapper(env, n_bins_obs=20, memory_size=200)

agent = RSUCBVIAgent(env, n_episodes=2000, gamma=0.99, horizon=200,
                     bonus_scale_factor=0.1, copy_env=False, min_dist=0.1)

agent.fit()

env.enable_rendering()
for ep in range(3):
    state = env.reset()
    for tt in range(agent.horizon):
        action = agent.policy(state)
        next_s, _, _, _ = env.step(action)
        state = next_s

# agent.env.render()
agent.env.plot(video_filename='test.mp4', n_skip=5, dot_scale_factor=15)
