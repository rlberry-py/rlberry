"""
===============================
A demo of rooms environment
===============================
 Illustration of NRooms environment

.. video:: ../../video_plot_rooms.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_rooms.jpg'

from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.agents.dynprog import ValueIterationAgent

env = NRoom(
    nrooms=9,
    remove_walls=False,
    room_size=9,
    initial_state_distribution="center",
    include_traps=True,
)
horizon = env.observation_space.n

agent = ValueIterationAgent(env, gamma=0.999, horizon=horizon)
print("fitting...")
info = agent.fit()
print(info)

env.enable_rendering()

for _ in range(10):
    state,info = env.reset()
    for tt in range(horizon):
        # action = agent.policy(state)
        action = env.action_space.sample()
        next_s, _, done, _ = env.step(action)
        if done:
            break
        state = next_s
env.render()
video = env.save_video("_video/video_plot_rooms.mp4")
