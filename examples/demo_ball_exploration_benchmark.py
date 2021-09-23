from rlberry.agents.mbqvi import MBQVIAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.wrappers.discretize_state import DiscretizeStateWrapper

env = get_benchmark_env(level=5)
env = DiscretizeStateWrapper(env, 25)

params = {}
params["n_samples"] = 20  # samples per state-action pair
params["gamma"] = 0.99
params["horizon"] = env.horizon

agent = MBQVIAgent(env, **params)
info = agent.fit()
print(info)

# evaluate policy in a deterministic version of the environment
env.enable_rendering()
state = env.reset()
for tt in range(5 * env.horizon):
    hh = tt
    if hh >= env.horizon:
        hh = tt % env.horizon
    action = agent.policy(state, hh)
    next_s, _, _, _ = env.step(action)
    state = next_s
env.render()
