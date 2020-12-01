from rlberry.agents import A2CAgent
from rlberry.envs.classic_control import MountainCar
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.seeding import seeding

render = True
seeding.set_global_seed(1223)

for env, n_episodes, horizon in zip([PBall2D(), MountainCar()],
                                    [400, 40000], [256, 512]):
    print("Running A2C on %s" % env.name)
    agent = A2CAgent(env, n_episodes=n_episodes, horizon=horizon,
                     gamma=0.99, learning_rate=0.001, k_epochs=4)
    agent.fit()

    if render:
        env.enable_rendering()
        state = env.reset()
        for tt in range(200):
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state

        env.render()
