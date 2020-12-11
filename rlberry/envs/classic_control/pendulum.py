"""
Pendulum environment adapted from OpenAI gym [1].

Modifications:
* render function follows the rlberry rendering interface

[1] https://github.com/openai/gym/blob/master/gym/
envs/classic_control/pendulum.py
"""

import numpy as np
import rlberry.spaces as spaces
from rlberry.envs.interface import Model
from rlberry.rendering import Scene, RenderInterface2D
from rlberry.rendering.common_shapes import bar_shape, circle_shape


class Pendulum(RenderInterface2D, Model):
    """
    The inverted pendulum swingup problem is a classic problem
    in the control literature. In this version of the problem,
    the pendulum starts in a random position, and the goal
    is to swing it up so it stays upright.
    """
    name = "Pendulum"

    def __init__(self):
        # init base classes
        Model.__init__(self)
        RenderInterface2D.__init__(self)

        # environment parameters
        self.max_speed = 8.
        self.max_torque = 2.
        self.dt = 0.5
        self.gravity = 10.
        self.mass = 1.
        self.length = 1.

        # rendering info
        self.set_clipping_area((-2.2, 2.2, -2.2, 2.2))
        self.set_refresh_interval(10)

        # observation and action spaces
        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        low = -high
        self.action_space = spaces.Box(low=-self.max_torque,
                                       high=self.max_torque,
                                       shape=(1,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # initialize
        self.reset()

    def reset(self):
        high = np.array([np.pi, 1])
        low = -high
        self.state = self.rng.uniform(low=low, high=high)
        self.last_action = None
        return self._get_ob()

    def step(self, action):
        assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(np.array(self.state))

        theta, thetadot = self.state
        gravity = self.gravity
        mass = self.mass
        length = self.length
        dt = self.dt

        action = np.clip(action, -self.max_torque, self.max_torque)[0]
        self.last_action = action # for rendering
        costs = angle_normalize(theta) ** 2 + .1 * thetadot ** 2 + .001 * (action ** 2)

        # compute the next state after action
        newthetadot = thetadot + (-3 * gravity / (2 * length) * np.sin(theta + np.pi) +
                                  3. / (mass * length ** 2) * action) * dt
        newtheta = theta + newthetadot * dt
        newthetadot = np.clip(newthetadot, -self.max_speed, self.max_speed)

        self.state = np.array([newtheta, newthetadot])
        return self._get_ob(), -costs, False, {}

    def _get_ob(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    #
    # Below code for rendering
    #

    def get_background(self):
        bg = Scene()
        return bg

    def get_scene(self, state):
        scene = Scene()

        p0 = (0.0, 0.0)
        p1 = (self.length * np.sin(state[0]),
              -self.length * np.cos(state[0]))

        link = bar_shape(p0, p1, 0.1)
        link.set_color((255/255, 105/255, 30/255))

        joint = circle_shape(p0, 0.075)
        joint.set_color((255/255, 215/255, 0/255))

        scene.add_shape(link)
        scene.add_shape(joint)

        return scene


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

