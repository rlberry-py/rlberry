from rlberry import agents
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry.agents.mbqvi import MBQVIAgent
from rlberry.wrappers.discretize_state import DiscretizeStateWrapper


env = TwinRooms()
env = DiscretizeStateWrapper(env, n_bins=20)
horizon = 25
agent = MBQVIAgent(env, n_samples=10, gamma=1.0, horizon=horizon)
agent.fit()

state = env.reset()
env.enable_rendering()
for ii in range(100):
    action = agent.policy(state)
    ns, rr, _, _ = env.step(action)
    state = ns

    if (ii+1) % horizon == 0:
        state = env.reset()

env.render()
