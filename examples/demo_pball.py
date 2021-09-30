import numpy as np
from rlberry.envs.benchmarks.ball_exploration import PBall2D

p = 5
A = np.array([
    [1.0, 0.1],
    [-0.1, 1.0]
]
)

reward_amplitudes = np.array([1.0, 0.5, 0.5])
reward_smoothness = np.array([0.25, 0.25, 0.25])

reward_centers = [
    np.array([0.75 * np.cos(np.pi / 2), 0.75 * np.sin(np.pi / 2)]),
    np.array([0.75 * np.cos(np.pi / 6), 0.75 * np.sin(np.pi / 6)]),
    np.array([0.75 * np.cos(5 * np.pi / 6), 0.75 * np.sin(5 * np.pi / 6)])
]

action_list = [0.1 * np.array([1, 0]),
               -0.1 * np.array([1, 0]),
               0.1 * np.array([0, 1]),
               -0.1 * np.array([0, 1])]

env = PBall2D(p=p, A=A,
              reward_amplitudes=reward_amplitudes,
              reward_centers=reward_centers,
              reward_smoothness=reward_smoothness,
              action_list=action_list)

env.enable_rendering()

for ii in range(100):
    env.step(1)
    env.step(3)

env.render()
env.save_video('pball.mp4')
