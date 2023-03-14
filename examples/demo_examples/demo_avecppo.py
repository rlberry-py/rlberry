"""
 =====================
 Demo: demo_avecppo
 =====================
"""
from rlberry.agents.experimental.torch import AVECPPOAgent
from rlberry.envs.benchmarks.ball_exploration import PBall2D


env = PBall2D()
n_episodes = 400
horizon = 256
agent = AVECPPOAgent(
    env, horizon=horizon, gamma=0.99, learning_rate=0.00025, eps_clip=0.2, k_epochs=4
)
agent.fit(budget=n_episodes)

env.enable_rendering()
observation,info = env.reset()
for tt in range(200):
    action = agent.policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.render()
