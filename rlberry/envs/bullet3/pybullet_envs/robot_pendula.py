import gym
import numpy as np

from rlberry.envs.bullet3.pybullet_envs.robot_bases import URDFBasedRobot2


class Pendulum(URDFBasedRobot2):
    swingup = False

    def __init__(self):
        # MJCFBasedRobot2.__init__(self, 'pendulum.xml', 'pole', action_dim=1, obs_dim=2)
        URDFBasedRobot2.__init__(self, "pendulum.urdf", "pole", action_dim=1, obs_dim=2)
        self.action_space = gym.spaces.Box(shape=(1,), low=-20, high=20)

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.pole = self.parts["pole"]
        self.j1 = self.jdict["hinge"]
        u = self.np_random.uniform(low=-0.1, high=0.1)
        self.j1.reset_current_position(u if not self.swingup else np.pi + u, 0)
        self.j1.set_motor_torque(0)

    def apply_action(self, a):
        assert np.isfinite(a).all()
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0
        self.j1.set_motor_torque(
            np.clip(a[0], self.action_space.low, self.action_space.high)
        )

    def calc_state(self):
        self.theta, theta_dot = self.j1.current_position()
        if not np.isfinite(self.theta):
            print("theta is inf")
            self.theta = 0

        if not np.isfinite(theta_dot):
            print("theta_dot is inf")
            theta_dot = 0

        return np.array([self.theta, theta_dot])


class PendulumSwingup(Pendulum):
    swingup = True
