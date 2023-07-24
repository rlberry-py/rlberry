import numpy as np
from gym import spaces
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.gym_pendulum_envs import InvertedPendulumBulletEnv
from pybullet_envs.scene_abstract import SingleRobotEmptyScene

from rlberry.envs.bullet3.pybullet_envs.robot_pendula import (Pendulum,
                                                              PendulumSwingup)


class PendulumBulletEnv(InvertedPendulumBulletEnv):
    """Simple pendulum"""

    def __init__(self):
        self.robot = Pendulum()
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(
            bullet_client, gravity=9.81, timestep=0.02, frame_skip=1
        )

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()  # sets self.pos_x self.pos_y
        if self.robot.swingup:
            reward = np.cos(self.robot.theta)
            done = False
        else:
            reward = 1.0
            done = np.abs(self.robot.theta) > 0.2
        self.rewards = [float(reward)]
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}


class PendulumSwingupBulletEnv(PendulumBulletEnv):
    def __init__(self):
        self.robot = PendulumSwingup()
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1


class DiscretePendulumBulletEnv(PendulumBulletEnv):
    """pybullet's InvertedPendulum with discrete actions"""

    def __init__(self):
        super().__init__()
        self.continuous_action_space = self.action_space
        self.action_space = spaces.Discrete(3)

    def step(self, a):
        if a == 0:
            return super().step(self.continuous_action_space.low)
        elif a == 1:
            return super().step(self.continuous_action_space.high)
        elif a == 2:
            return super().step(np.zeros(self.continuous_action_space.shape))
        else:
            raise IndexError


class DiscretePendulumSwingupBulletEnv(PendulumSwingupBulletEnv):
    """pybullet's InvertedPendulumSwingup with discrete actions"""

    def __init__(self):
        super().__init__()
        self.continuous_action_space = self.action_space
        self.action_space = spaces.Discrete(3)

    def step(self, a):
        if a == 0:
            return super().step(self.continuous_action_space.low)
        elif a == 1:
            return super().step(self.continuous_action_space.high)
        elif a == 2:
            return super().step(np.zeros(self.continuous_action_space.shape))
        else:
            raise IndexError
