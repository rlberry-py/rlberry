""" 
 ===================== 
 Demo: demo_value_iteration 
 =====================
"""
from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.envs.finite import GridWorld, Chain

for env in [Chain(), GridWorld(7, 10, walls=((2, 2), (3, 3)))]:
    agent = ValueIterationAgent(env, gamma=0.95)
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
