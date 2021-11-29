from rlberry.agents.torch import PPOAgent
from rlberry.envs.benchmarks.ball_exploration import PBall2D


env = PBall2D()
n_episodes = 200
horizon = 256

agent = PPOAgent(
    env,
    horizon=horizon,
    gamma=0.99,
    learning_rate=0.001,
    eps_clip=0.2,
    k_epochs=4)
agent.fit(budget=n_episodes)

env.enable_rendering()
state = env.reset()
for tt in range(200):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

env.render()
