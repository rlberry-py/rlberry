""" 
 ===================== 
 Demo: demo_vis2d 
 =====================
"""
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom, get_nroom_state_coord
from rlberry.envs.classic_control import MountainCar
from rlberry.wrappers.vis2d import Vis2dWrapper
from rlberry.agents import RSUCBVIAgent
from rlberry.agents.dynprog import ValueIterationAgent

CHOICE = 1

if CHOICE == 0:
    env = NRoom(nrooms=5, array_observation=False, reward_free=True)
    env = Vis2dWrapper(env, n_bins_obs=20, memory_size=100, state_preprocess_fn=get_nroom_state_coord)
    agent = ValueIterationAgent(env.unwrapped, gamma=0.99, horizon=200, copy_env=False)

else:
    env = MountainCar()
    env = Vis2dWrapper(env, n_bins_obs=20, memory_size=200)

    agent = RSUCBVIAgent(env, gamma=0.99, horizon=200,
                         bonus_scale_factor=0.1, copy_env=False, min_dist=0.1)

agent.fit(budget=100)

env.enable_rendering()
for ep in range(3):
    state = env.reset()
    for tt in range(agent.horizon):
        action = agent.policy(state)
        next_s, _, _, _ = env.step(action)
        state = next_s

try:
    xlim = [env.observation_space.low[0], env.observation_space.high[0]]
    ylim = [env.observation_space.low[1], env.observation_space.high[1]]
except Exception:
    xlim = None
    ylim = None

# env.render()
env.plot_trajectories(n_skip=5, dot_scale_factor=15, xlim=xlim, ylim=ylim, dot_size_means='total_visits')
env.plot_trajectory_actions(xlim=xlim, ylim=ylim)
