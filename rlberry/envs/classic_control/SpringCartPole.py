"""
SpringCartPole environment introduced in J-F. Hren PhD thesis.
"""

import numpy as np
import rlberry.spaces as spaces
from rlberry.envs.interface import Model
from rlberry.rendering import Scene, GeometricPrimitive, RenderInterface2D
from rlberry.rendering.common_shapes import bar_shape, circle_shape


class SpringCartPole(RenderInterface2D, Model):
    """
    SpringCartPole is an extension of the CartPole environment proposed in
    PhD thesis J-F. Hren. It consists of two carts connected by a spring.

    Parameters
    ----------
    dt : float, defualt=0.02
        Time step of the simulation.
    obs_trans : bool, default=True
        If True, state has dimension 10:
        State:
            'Cart position1', 'Cart velocity1', 'Pole cos1', 'Pole sin1', 'Pole angular velocity1',
            'Cart position2', 'Cart velocity2', 'Pole cos2', 'Pole sin2', 'Pole angular velocity2'
        If False, state has dimension 8:
        State:
            'Cart position1', 'Cart velocity1', 'Pole angle1', 'Pole angular velocity1',
            'Cart position2', 'Cart velocity2', 'Pole angle2', 'Pole angular velocity2'
    swing_up : bool, default=False
        If True, the pole starting position is at the bottom
        If False, the pole starting position is at the top
    random_init : bool, default=True
        If True, the noise is added to the carts and poles starting positions

    Notes
    -----
    State:
        The state consists of the position of cart 1, its speed, the angle
        of pole 1 (expressed in radians or in a tuple of cos() and sin()) and
        its angular speed, and the same set of values for cart 2 and pole 2.
        For both poles, the angle of 0 corresponds to the vertical position,
        the positive angles correspond to a counterclockwise rotation.

    Actions:
        The action is either 0, 1, 2, or 3, corresponding to the four possible
        actions:
            LL = 0, move cart 1 to the left, cart 2 to the left
            RR = 1, move cart 1 to the right, cart 2 to the right
            LR = 2, move cart 1 to the left, cart 2 to the right
            RL = 3, move cart 1 to the right, cart 2 to the left
        The magnitude of actions is fixed to 2.0.

    Reference:
    .. seealso::
        J-F. Hren: Planification optimiste pour systèmes déterministes, PhD thesis
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic than Euler method
    """

    name = "SpringCartPole"

    ACT_RNG = 2.0
    AVAIL_TORQUES = [
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
        np.array([-2.0, 2.0]),
        np.array([2.0, -2.0]),
    ]

    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 4

    def __init__(self, dt=0.02, obs_trans=True, swing_up=False, random_init=True):
        Model.__init__(self)
        RenderInterface2D.__init__(self)

        self.dt = dt
        self.gravity = 9.81
        self.track_length = 2.0
        self.L = 0.5 * self.track_length
        self.pole_length = 1.0
        self.l = 0.5 * self.pole_length
        self.masspole = 0.1
        self.masscart = 1.0
        self.cart_friction = 5e-4
        self.pole_friction = 2e-6
        self.spring = 2.0
        self.normal_spring_length = 0.5
        self.min_spring_length = 0.1
        self.max_spring_length = 1.5
        self.max_velocity = 15.0
        self.ang_velocity = 10.0
        self.force_mag = self.ACT_RNG
        self.swing_up = swing_up
        self.random_init = random_init
        self.obs_trans = obs_trans

        if self.obs_trans:
            self.obs_shape = 10
        else:
            self.obs_shape = 8

        # init base classes
        self.reward_range = (0.0, 1.0)

        # rendering info
        boundy = self.pole_length * 2 + 0.2
        boundx = self.track_length + self.pole_length * 2 + 0.2
        # (left, right, bottom, top)
        self.set_clipping_area((-boundx, boundx, -boundy, boundy))
        self.set_refresh_interval(10)  # in milliseconds

        # observation and action spaces
        if self.obs_trans:
            high = np.array(
                [
                    self.track_length,
                    np.finfo(np.float32).max,
                    1,
                    1,
                    np.finfo(np.float32).max,
                    self.track_length,
                    np.finfo(np.float32).max,
                    1,
                    1,
                    np.finfo(np.float32).max,
                ]
            )
        else:
            high = np.array(
                [
                    self.track_length,
                    np.finfo(np.float32).max,
                    2 * np.pi,
                    np.finfo(np.float32).max,
                    self.track_length,
                    np.finfo(np.float32).max,
                    2 * np.pi,
                    np.finfo(np.float32).max,
                ]
            )
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(4)

        # initialize
        self.state = None  # state in pos or angles
        self.state_ = None  # state in angles
        self.reset()

    def transform_states(self, state):
        """Transform state with dim=8 to the state with dim=10"""
        assert state.shape[-1] == 8, "State has wrong shape, should be 8"
        shape = list(state.shape)
        shape[-1] = 10
        state_ = np.zeros(shape)
        state_[..., :2] = state[..., :2]
        state_[..., 4:7] = state[..., 3:6]
        state_[..., -1] = state[..., -1]
        theta1 = state[..., 2]
        theta2 = state[..., 6]
        state_[..., 2] = np.cos(theta1)
        state_[..., 3] = np.sin(theta1)
        state_[..., 7] = np.cos(theta2)
        state_[..., 8] = np.sin(theta2)
        return state_

    # def trigonometric2angle(self, costheta, sintheta):
    #     C = costheta**2 + sintheta**2
    #     costheta, sintheta = costheta / C, sintheta / C
    #     theta = np.arctan2(sintheta / C, costheta / C)
    #     return theta

    def reset(self):
        if self.random_init:
            rand_state = self.rng.uniform(low=-0.1, high=0.1, size=(8,))
        else:
            rand_state = np.zeros((8,))
        rand_state[4] += self.normal_spring_length
        if self.swing_up:
            rand_state[2] += np.pi
            rand_state[6] += np.pi
        if self.obs_trans:
            self.state = self.transform_states(rand_state)
        else:
            self.state = rand_state
        self.state_ = rand_state
        return self.state

    def _reward(self):
        state = self.state
        if state.shape[-1] == 10:
            (
                _,
                _,
                cos1,
                sin1,
                _,
                _,
                _,
                cos2,
                sin2,
                _,
            ) = np.split(state, 10, axis=-1)
            C1 = np.sqrt(cos1**2 + sin1**2)
            C2 = np.sqrt(cos2**2 + sin2**2)
            cos1 = cos1 / C1
            sin1 = sin1 / C1
            cos2 = cos2 / C2
            sin2 = sin2 / C2
        else:
            _, _, theta1, _, _, _, theta2, _ = np.split(state, 8, axis=-1)
            cos1 = np.cos(theta1)
            sin1 = np.sin(theta1)
            cos2 = np.cos(theta2)
            sin2 = np.sin(theta2)

        bad_condition = self._terminal()

        pos_reward = (1 + cos1) / 4 + (1 + cos2) / 4
        neg_reward = 0.0

        return np.where(bad_condition, neg_reward, pos_reward)

    def bound_states(self, state):
        assert state.shape[-1] == 8, "state must be of shape (8,)"
        x1, x1dot, theta1, theta1dot, x2, x2dot, theta2, theta2dot = np.split(
            state, 8, axis=-1
        )
        theta1 = np.asarray(wrap(theta1, -np.pi, np.pi))
        theta2 = np.asarray(wrap(theta2, -np.pi, np.pi))
        x1dot = np.asarray(bound(x1dot, [-self.max_velocity, self.max_velocity]))
        x2dot = np.asarray(bound(x2dot, -self.max_velocity, self.max_velocity))
        theta1dot = np.asarray(bound(theta1dot, -self.ang_velocity, self.ang_velocity))
        theta2dot = np.asarray(bound(theta2dot, -self.max_velocity, self.ang_velocity))
        state = np.concatenate(
            [x1, x1dot, theta1, theta1dot, x2, x2dot, theta2, theta2dot], axis=-1
        )
        return state

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(np.array(self.state_))

        s = self.state_
        torque = self.AVAIL_TORQUES[action]

        # # Add noise to the force action
        # if self.torque_noise_max > 0:
        #     torque += self.rng.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:-2]  # omit action

        ns = self.bound_states(ns)
        self.state_ = ns
        if self.obs_trans:
            self.state = self.transform_states(ns)
        else:
            self.state = ns
        terminal = self._terminal()
        reward = self._reward()[0]
        return self.state, reward, terminal, {}

    def _terminal(self):
        s = self.state_
        x1 = s[0]
        x2 = s[4]
        bad_condition = False
        bad_condition += np.abs(x1) > self.L
        bad_condition += np.abs(x2) > self.L
        bad_condition += x2 <= x1
        bad_condition += np.abs(x1 - x2) < self.min_spring_length
        bad_condition += np.abs(x1 - x2) > self.max_spring_length

        return bool(bad_condition)

    def _dsdt(self, sa, t):
        assert sa.shape[-1] == 10, "state + action must be of shape (10,)"
        x1, x1dot, theta1, theta1dot, x2, x2dot, theta2, theta2dot, a1, a2 = np.split(
            sa, 10, axis=-1
        )
        cos1 = np.cos(theta1)
        sin1 = np.sin(theta1)
        cos2 = np.cos(theta2)
        sin2 = np.sin(theta2)
        # x1 - size [N, 1] or [L, N, 1]

        f1 = a1 + self.spring * (self.normal_spring_length - np.abs(x1 - x2))
        f2 = a2 + self.spring * (self.normal_spring_length - np.abs(x1 - x2))

        a11 = 4 * self.l / 3
        a22 = -self.masscart - self.masspole

        a121 = -cos1
        a122 = -cos2
        a211 = self.l * self.masspole * cos1
        a212 = self.l * self.masspole * cos2

        b11 = (
            self.gravity * sin1
            - self.pole_friction * theta1dot / self.l / self.masspole
        )
        b12 = (
            self.gravity * sin2
            - self.pole_friction * theta2dot / self.l / self.masspole
        )

        b21 = (
            self.l * self.masspole * sin1 * theta1dot**2
            - f1
            + self.cart_friction * np.sign(x1dot)
        )
        b22 = (
            self.l * self.masspole * sin2 * theta2dot**2
            - f2
            + self.cart_friction * np.sign(x2dot)
        )

        theta1acc = (a121 * b21 - a22 * b11) / (a121 * a211 - a11 * a22)
        theta2acc = (a122 * b22 - a22 * b12) / (a122 * a212 - a11 * a22)

        x1acc = (b11 - a11 * theta1acc) / a121
        x2acc = (b12 - a11 * theta2acc) / a122

        a1dot = np.zeros_like(a1)
        a2dot = np.zeros_like(a2)

        return np.concatenate(
            [
                x1dot,
                x1acc,
                theta1dot,
                theta1acc,
                x2dot,
                x2acc,
                theta2dot,
                theta2acc,
                a1dot,
                a2dot,
            ],
            axis=-1,
        )

    #
    # Below: code for rendering
    #

    def get_background(self):
        bg = Scene()
        return bg

    def get_scene(self, state):
        scene = Scene()
        SCALE = 3

        assert state.shape[-1] == 8, "state must be of shape (8,)"

        x1 = state[0]
        x2 = state[4]
        theta1 = state[2]
        theta2 = state[6]

        cartx1 = x1 * SCALE  # MIDDLE OF CART 1

        cartx2 = x2 * SCALE  # MIDDLE OF CART 2

        cartwidth = 0.05 * SCALE

        c1p1 = (
            cartx1 - cartwidth / 2,
            0,
        )
        c1p2 = (
            cartx1 + cartwidth / 2,
            0,
        )

        c2p1 = (
            cartx2 - cartwidth / 2,
            0,
        )
        c2p2 = (
            cartx2 + cartwidth / 2,
            0,
        )

        p1 = (
            cartx1 - np.sin(theta1) * self.pole_length * SCALE,
            np.cos(theta1) * self.pole_length * SCALE,
        )

        p01 = (cartx1, 0)
        p02 = (cartx2, 0)

        p2 = (
            cartx2 - np.sin(theta2) * self.pole_length * SCALE,
            np.cos(theta2) * self.pole_length * SCALE,
        )

        cart1 = bar_shape(c1p1, c1p2, 0.02 * SCALE)
        cart1.set_color((255 / 255, 100 / 255, 0 / 255))

        cart2 = bar_shape(c2p1, c2p2, 0.02 * SCALE)
        cart2.set_color((255 / 255, 100 / 255, 0 / 255))

        pole1 = bar_shape(p01, p1, 0.01 * SCALE)
        pole1.set_color((255 / 255, 215 / 255, 0 / 255))

        pole2 = bar_shape(p02, p2, 0.01 * SCALE)
        pole2.set_color((255 / 255, 215 / 255, 0 / 255))

        spring = bar_shape(
            p01,
            p02,
            0.03
            * np.sqrt(self.normal_spring_length)
            * SCALE
            / np.sqrt(cartx2 - cartx1),
        )
        spring.set_color((50 / 255, 50 / 255, 50 / 255))

        joint1 = circle_shape(p01, 0.03)
        joint1.set_color((0 / 255, 255 / 255, 0 / 255))

        joint2 = circle_shape(p02, 0.03)
        joint2.set_color((0 / 255, 255 / 255, 0 / 255))

        track_line = GeometricPrimitive("LINES")
        track_line.add_vertex((-self.track_length / 2 * SCALE, -0.02 * SCALE))
        track_line.add_vertex((self.track_length / 2 * SCALE, -0.02 * SCALE))

        axis1 = GeometricPrimitive("LINES")
        axis1.add_vertex((cartx1, 0))
        axis1.add_vertex((cartx1, self.pole_length * SCALE))
        axis1.set_color((250 / 255, 250 / 255, 250 / 255))

        axis2 = GeometricPrimitive("LINES")
        axis2.add_vertex((cartx2, 0))
        axis2.add_vertex((cartx2, self.pole_length * SCALE))
        axis2.set_color((250 / 255, 250 / 255, 250 / 255))

        scene.add_shape(cart1)
        scene.add_shape(cart2)
        scene.add_shape(pole1)
        scene.add_shape(pole2)
        scene.add_shape(joint1)
        scene.add_shape(joint2)
        scene.add_shape(spring)
        scene.add_shape(track_line)

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
    return np.clip(x, m, M)


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
