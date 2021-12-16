""" 
 ===================== 
 Demo: demo_a2c 
 =====================
"""
from rlberry.agents.torch import A2CAgent
from rlberry.envs.benchmarks.ball_exploration import PBall2D


env = PBall2D()
n_episodes = 400
horizon = 256
agent = A2CAgent(
    env,
    horizon=horizon,
    gamma=0.99,
    learning_rate=0.001,
    k_epochs=4)
agent.fit(budget=n_episodes)


env.enable_rendering()
state = env.reset()
for tt in range(200):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

env.render()
