"""
Acrobot environment adapted from OpenAI gym [1].

Modifications:
* define reward_range
* render function follows the rlberry rendering interface.

[1] https://github.com/openai/gym/blob/master/gym/
envs/classic_control/acrobot.py
"""

import numpy as np
import rlberry.spaces as spaces
from rlberry.envs.interface import Model
from rlberry.rendering import Scene, GeometricPrimitive, RenderInterface2D
from rlberry.rendering.common_shapes import bar_shape, circle_shape

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"


# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py


class Acrobot(RenderInterface2D, Model):
    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.

    Notes
    -----
    State:
        The state consists of the sin() and cos() of the two rotational joint
        angles and the joint angular velocities:
        [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
        For the first link, an angle of 0 corresponds to the link pointing
        downwards.
        The angle of the second link is relative to the angle of the first link.
        An angle of 0 corresponds to having the same angle between the two links.
        A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.

    Actions:
        The action is either applying +1, 0 or -1 torque on the joint between
        the two pendulum links.
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'

    Reference:
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """
    name = "Acrobot"

    dt = .2

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    AVAIL_TORQUE = [-1., 0., +1]

    torque_noise_max = 0.

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self):
        # init base classes
        Model.__init__(self)
        RenderInterface2D.__init__(self)
        self.reward_range = (-1.0, 0.0)

        # rendering info
        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2
        # (left, right, bottom, top)
        self.set_clipping_area((-bound, bound, -bound, bound))
        self.set_refresh_interval(10)  # in milliseconds

        # observation and action spaces
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(3)

        # initialize
        self.state = None
        self.reset()

    def reset(self):
        self.state = self.rng.uniform(low=-0.1, high=0.1, size=(4,))
        return self._get_ob()

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(np.array(self.state))

        s = self.state
        torque = self.AVAIL_TORQUE[action]

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.rng.uniform(-self.torque_noise_max,
                                       self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous,
        # [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        return self._get_ob(), reward, terminal, {}

    def _get_ob(self):
        s = self.state
        return np.array([np.cos(s[0]), np.sin(s[0]), np.cos(s[1]),
                         np.sin(s[1]), s[2], s[3]])

    def _terminal(self):
        s = self.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
             (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2) \
               + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
                       (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation
            # and the book
            ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 *
                        np.sin(theta2) - phi2) \
                       / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    #
    # Below: code for rendering
    #

    def get_background(self):
        bg = Scene()
        return bg

    def get_scene(self, state):
        scene = Scene()

        p0 = (0.0, 0.0)

        p1 = (self.LINK_LENGTH_1 * np.sin(state[0]),
              -self.LINK_LENGTH_1 * np.cos(state[0]))
        p2 = (p1[0] + self.LINK_LENGTH_2 * np.sin(state[0] + state[1]),
              p1[1] - self.LINK_LENGTH_2 * np.cos(state[0] + state[1]))

        link1 = bar_shape(p0, p1, 0.1)
        link1.set_color((255 / 255, 140 / 255, 0 / 255))

        link2 = bar_shape(p1, p2, 0.1)
        link2.set_color((210 / 255, 105 / 255, 30 / 255))

        joint1 = circle_shape(p0, 0.075)
        joint1.set_color((255 / 255, 215 / 255, 0 / 255))

        joint2 = circle_shape(p1, 0.075)
        joint2.set_color((255 / 255, 215 / 255, 0 / 255))

        goal_line = GeometricPrimitive("LINES")
        goal_line.add_vertex((-5, 1))
        goal_line.add_vertex((5, 1))

        scene.add_shape(link1)
        scene.add_shape(link2)
        scene.add_shape(joint1)
        scene.add_shape(joint2)
        scene.add_shape(goal_line)

        return scene


def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined
    by m, M.
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Parameters
    ----------
        x: a scalar
        m:
            minimum possible value in range
        M:
            maximum possible value in range

    Returns
    -------
        x:
            a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    Parameters
    ----------
    x:
        scalar

    Returns
    -------
    x:
        scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    Parameters:
    -----------
    derivs:
        the derivative of the system and has the signature
        ``dy = derivs(yi, ti)``
    y0:
        initial state vector
    t:
        sample times
    args:
        additional arguments passed to the derivative function
    kwargs:
        additional keyword arguments passed to the derivative function

    Returns
    -------
    yout:
        Runge-Kutta approximation of the ODE

    Examples
    --------
    Example 1::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)

    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
