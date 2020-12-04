from rlberry.agents.cem import CEMAgent
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
import rlberry.seeding as seeding

seeding.set_global_seed(1234)


env = get_benchmark_env(level=1)

n_episodes = 1000
horizon = 30
gamma = 0.99


params = {
    'n_episodes': n_episodes,
    'horizon': horizon,
    'gamma': gamma,
    'batch_size': 20,
    'percentile': 70,
    'learning_rate': 0.01
}

agent = CEMAgent(env, **params)
agent.fit()

env.enable_rendering()
state = env.reset()
for tt in range(4*horizon):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

env.render()
