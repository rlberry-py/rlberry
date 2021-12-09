""" 
 ===================== 
 Demo: demo_mbqvi 
 =====================
"""
from rlberry.agents.mbqvi import MBQVIAgent
from rlberry.envs.finite import GridWorld

params = {}
params["n_samples"] = 100  # samples per state-action pair
params["gamma"] = 0.99
params["horizon"] = None

env = GridWorld(7, 10, walls=((2, 2), (3, 3)), success_probability=0.6)
agent = MBQVIAgent(env, **params)
info = agent.fit()
print(info)

# evaluate policy in a deterministic version of the environment
env_eval = GridWorld(7, 10, walls=((2, 2), (3, 3)), success_probability=1.0)
env_eval.enable_rendering()
state = env_eval.reset()
for tt in range(50):
    action = agent.policy(state)
    next_s, _, _, _ = env_eval.step(action)
    state = next_s
env_eval.render()
