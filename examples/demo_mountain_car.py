from rlberry.envs.classic_control import MountainCar

env = MountainCar()
env.enable_rendering()
for tt in range(150):
    env.step(env.action_space.sample())
env.render()
