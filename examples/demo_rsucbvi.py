from rlberry.agents import RSUCBVIAgent
from rlberry.envs.classic_control import MountainCar
from rlberry.envs.benchmarks.ball_exploration import PBall2D

for env, horizon in zip([MountainCar(), PBall2D()], [170, 50]):
    print("Running RS-UCBVI on %s" % env.name)
    agent = RSUCBVIAgent(env, n_episodes=1000, gamma=0.99, horizon=horizon,
                         bonus_scale_factor=0.1)
    agent.fit()

    env.enable_rendering()
    state = env.reset()
    for tt in range(200):
        action = agent.policy(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state

    env.render()
