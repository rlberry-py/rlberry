from rlberry.envs.finite  import GridWorld
from rlberry.agents.dynprog import ValueIterationAgent

env = GridWorld(7, 10, walls=((2,2), (3,3)))
agent = ValueIterationAgent(env, gamma=0.95)
agent.fit()

env.enable_rendering()

state = env.reset()
for tt in range(50):
    action = agent.policy(state)
    next_s, _, _, _ = env.step(action)
    state = next_s

env.render()