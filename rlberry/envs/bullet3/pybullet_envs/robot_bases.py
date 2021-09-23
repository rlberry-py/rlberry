import os
import pybullet
from pybullet_envs.robot_bases import MJCFBasedRobot, URDFBasedRobot

# Use our custom data
from rlberry.envs.bullet3 import data


class MJCFBasedRobot2(MJCFBasedRobot):
    def reset(self, bullet_client):
        self._p = bullet_client
        # print("Created bullet_client with id=", self._p._client)
        if (self.doneLoading == 0):
            self.ordered_joints = []
            self.doneLoading = 1
            if self.self_collision:
                self.objects = self._p.loadMJCF(os.path.join(data.getDataPath(), "mjcf",
                                                             self.model_xml),
                                                flags=pybullet.URDF_USE_SELF_COLLISION |
                                                      pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                                                      pybullet.URDF_GOOGLEY_UNDEFINED_COLORS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)
            else:
                self.objects = self._p.loadMJCF(
                    os.path.join(data.getDataPath(), "mjcf", self.model_xml,
                                 flags=pybullet.URDF_GOOGLEY_UNDEFINED_COLORS))
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)
        self.robot_specific_reset(self._p)

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return s


class URDFBasedRobot2(URDFBasedRobot):
    def __init__(self,
                 model_urdf,
                 robot_name,
                 action_dim,
                 obs_dim,
                 basePosition=[0, 0, 0],
                 baseOrientation=[0, 0, 0, 1],
                 fixed_base=False,
                 self_collision=False):
        super().__init__(model_urdf, robot_name, action_dim, obs_dim, basePosition, baseOrientation, fixed_base,
                         self_collision)
        self.doneLoading = 0

    def reset(self, bullet_client):
        self._p = bullet_client
        if (self.doneLoading == 0):
            self.ordered_joints = []
            self.doneLoading = 1
            if self.self_collision:
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p,
                    self._p.loadURDF(os.path.join(data.getDataPath(), self.model_urdf),
                                     basePosition=self.basePosition,
                                     baseOrientation=self.baseOrientation,
                                     useFixedBase=self.fixed_base,
                                     flags=pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_GOOGLEY_UNDEFINED_COLORS))
            else:
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p,
                    self._p.loadURDF(os.path.join(data.getDataPath(), self.model_urdf),
                                     basePosition=self.basePosition,
                                     baseOrientation=self.baseOrientation,
                                     useFixedBase=self.fixed_base, flags=pybullet.URDF_GOOGLEY_UNDEFINED_COLORS))

        self.robot_specific_reset(self._p)

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.potential = self.calc_potential()

        return s
