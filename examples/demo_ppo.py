from rlberry.agents.torch import PPOAgent
from rlberry.envs.classic_control import MountainCar
from rlberry.envs.benchmarks.ball_exploration import PBall2D

render = False

for env, n_episodes, horizon in zip([PBall2D(), MountainCar()],
                                    [400, 40000], [256, 512]):
    print(f'Running PPO on {env.name}')
    agent = PPOAgent(env,
                     n_episodes=n_episodes,
                     horizon=horizon,
                     gamma=0.99,
                     learning_rate=0.001,
                     eps_clip=0.2,
                     k_epochs=4)
    agent.fit()

    if render:
        env.enable_rendering()
        state = env.reset()
        for tt in range(200):
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state

        env.render()
