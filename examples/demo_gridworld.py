from rlberry.envs.finite import GridWorld

env = GridWorld(7, 10, walls=((2, 2), (3, 3)))
env.enable_rendering()
for tt in range(50):
    env.step(env.action_space.sample())
env.render()
