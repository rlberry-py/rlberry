import gymnasium as gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# ------------bullet-------------

register(
    id="PendulumBulletEnv-v0",
    entry_point="rlberry.envs.bullet3.pybullet_envs.gym_pendulum_envs:PendulumBulletEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="PendulumSwingupBulletEnv-v0",
    entry_point="rlberry.envs.bullet3.pybullet_envs.gym_pendulum_envs:PendulumSwingupBulletEnv",
    max_episode_steps=1000,
    reward_threshold=800.0,
)

register(
    id="DiscretePendulumBulletEnv-v0",
    entry_point="rlberry.envs.bullet3.pybullet_envs.gym_pendulum_envs:DiscretePendulumBulletEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="DiscretePendulumSwingupBulletEnv-v0",
    entry_point="rlberry.envs.bullet3.pybullet_envs.gym_pendulum_envs:DiscretePendulumSwingupBulletEnv",
    max_episode_steps=1000,
    reward_threshold=800.0,
)
