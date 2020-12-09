from rlberry.envs.benchmarks.grid_exploration.four_room import FourRoom
from rlberry.agents.dynprog import ValueIterationAgent


env = FourRoom(reward_free=True, difficulty=0, array_observation=False)

agent = ValueIterationAgent(env, gamma=0.99)
info = agent.fit()
print(info)

env.enable_rendering()

state = env.reset()
for tt in range(50):
    action = agent.policy(state)
    next_s, _, done, _ = env.step(action)
    if done:
        break
    state = next_s
env.render()
