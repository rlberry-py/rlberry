from rlberry.envs.benchmarks.grid_exploration.four_room import FourRoom
from rlberry.envs.benchmarks.grid_exploration.six_room import SixRoom
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.agents.dynprog import ValueIterationAgent


# env = FourRoom(reward_free=False, difficulty=0, array_observation=False)
# env = SixRoom(reward_free=False, array_observation=False)
env = NRoom(nrooms=9,
            remove_walls=False,
            room_size=9,
            initial_state_distribution='center',
            include_traps=True)
horizon = env.observation_space.n

agent = ValueIterationAgent(env, gamma=0.999, horizon=horizon)
print("fitting...")
info = agent.fit()
print(info)

env.enable_rendering()

for _ in range(10):
    state = env.reset()
    for tt in range(horizon):
        # action = agent.policy(state)
        action = env.action_space.sample()
        next_s, _, done, _ = env.step(action)
        if done:
            break
        state = next_s
env.render()
