from rlberry.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry.agents.dynprog import ValueIterationAgent


env = AppleGold(reward_free=False, array_observation=False)

agent = ValueIterationAgent(env, gamma=0.9)
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

