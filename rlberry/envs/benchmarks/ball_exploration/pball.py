import numpy as np
import logging

import rlberry.spaces as spaces
from rlberry.envs.interface import Model
from rlberry.rendering import Scene, GeometricPrimitive, RenderInterface2D

logger = logging.getLogger(__name__)


def projection_to_pball(x, p):
    """
    Solve the problem:
        min_z  ||x-z||_2^2
        s.t.   ||z||_p  <= 1
    for p = 2 or p = np.inf

    If p is not 2 or np.inf, it returns x/norm_p(x) if norm_p(x) > 1

    WARNING: projection_to_pball is not actually a projection for p!=2
    or p=!np.inf
    """
    if np.linalg.norm(x, ord=p) <= 1.0:
        return x

    if p == 2:
        z = x / np.linalg.norm(x, ord=p)
        return z

    if p == np.inf:
        z = np.minimum(1.0, np.maximum(x, -1.0))
        return z

        # below it is not a projection
    return x / np.linalg.norm(x, ord=p)


class PBall(Model):
    """
    Parametric family of environments whose state space is a unit sphere
    according to the p-norm in R^d.

    Note:
        The projection function is only a true projection for
        p in {2, infinity}.

    ----------------------------------------------------------------------
    State space:
        x in R^d: norm_p (x) <= 1

        implemented as rlberry.spaces.Box representing [0, 1]^d
    ----------------------------------------------------------------------
    Action space:
        {u_1, ..., u_m} such that u_i in R^d'  for i = 1, ..., m

        implemented as rlberry.spaces.Discrete(m)
    ----------------------------------------------------------------------
    Reward function (independent of the actions):
        r(x) = sum_{i=1}^n  b_i  max( 0,  1 - norm_p( x - x_i )/c_i )

        requirements:
            c_i >= 0
            b_i in [0, 1]
    ----------------------------------------------------------------------
    Transitions:
        x_{t+1} = A x_t + B u_t + N

        where
            A: square matrix of size d
            B: matrix of size (d, d')
            N: d-dimensional Gaussian noise with zero mean and covariance
            matrix sigma*I
    ----------------------------------------------------------------------
    Initial state:
        d-dimensional Gaussian with mean mu_init and covariance matrix
        sigma_init*I
    ----------------------------------------------------------------------

    Default parameters are provided for a 2D environment, PBall2D
    """

    name = "LP-Ball"

    def __init__(self,
                 p,
                 action_list,
                 reward_amplitudes,
                 reward_smoothness,
                 reward_centers,
                 A,
                 B,
                 sigma,
                 sigma_init,
                 mu_init):
        """
        Parameters
        -----------
        p : int
            parameter of the p-norm
        action_list : list
            list of actions {u_1, ..., u_m}, each action u_i is a
            d'-dimensional array
        reward_amplitudes: list
            list of reward amplitudes: {b_1, ..., b_n}
        reward_smoothness : list
            list of reward smoothness: {c_1, ..., c_n}
        reward_centers : list
            list of reward centers:    {x_1, ..., x_n}
        A : numpy.ndarray
            array A of size (d, d)
        B : numpy.ndarray
            array B of size (d, d')
        sigma : double
            transition noise sigma
        sigma_init : double
            initial state noise sigma_init
        mu_init : numpy.ndarray
            array of size (d,) containing the mean of the initial state
        """
        Model.__init__(self)

        assert p >= 1, "PBall requires p>=1"
        if p not in [2, np.inf]:
            logger.warning("For p!=2 or p!=np.inf, PBall \
does not make true projections onto the lp ball.")
        self.p = p
        self.d, self.dp = B.shape  # d and d'
        self.m = len(action_list)
        self.action_list = action_list
        self.reward_amplitudes = reward_amplitudes
        self.reward_smoothness = reward_smoothness
        self.reward_centers = reward_centers
        self.A = A
        self.B = B
        self.sigma = sigma
        self.sigma_init = sigma_init
        self.mu_init = mu_init

        # State and action spaces
        low = -1.0 * np.ones(self.d, dtype=np.float64)
        high = np.ones(self.d, dtype=np.float64)
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(self.m)

        # reward range
        assert len(self.reward_amplitudes) == len(self.reward_smoothness)
        assert len(self.reward_amplitudes) == len(self.reward_centers)
        if len(self.reward_amplitudes) > 0:
            assert self.reward_amplitudes.max() <= 1.0 and \
                   self.reward_amplitudes.min() >= 0.0, \
                "reward amplitudes b_i must be in [0, 1]"
            assert self.reward_smoothness.min() > 0.0, \
                "reward smoothness c_i must be > 0"
        self.reward_range = (0, 1.0)

        #
        self.name = "Lp-Ball"

        # Initalize state
        self.reset()

    def reset(self, state=None):
        if state is not None:
            self.state = state
        else:
            self.state = self.mu_init \
                         + self.sigma_init * self.seeder.rng.normal(size=self.d)
            # projection to unit ball
        self.state = projection_to_pball(self.state, self.p)
        return self.state.copy()

    def sample(self, state, action):
        assert self.action_space.contains(action)
        assert self.observation_space.contains(state)

        # next state
        action_vec = self.action_list[action]
        next_s = self.A.dot(state) + self.B.dot(action_vec) \
                 + self.sigma * self.rng.normal(size=self.d)
        next_s = projection_to_pball(next_s, self.p)

        # done and reward
        done = False
        reward = self.compute_reward_at(state)

        return next_s, reward, done, {}

    def step(self, action):
        next_s, reward, done, info = self.sample(self.state, action)
        self.state = next_s.copy()
        return next_s, reward, done, info

    def compute_reward_at(self, x):
        reward = 0.0
        for ii, b_ii in enumerate(self.reward_amplitudes):
            c_ii = self.reward_smoothness[ii]
            x_ii = self.reward_centers[ii]
            dist = np.linalg.norm(x - x_ii, ord=self.p)
            reward += b_ii * max(0.0, 1.0 - dist / c_ii)
        return reward

    def get_reward_lipschitz_constant(self):
        ratios = self.reward_amplitudes / self.reward_smoothness
        Lr = ratios.max()
        return Lr

    def get_transitions_lipschitz_constant(self):
        """
        note: considers a fixed action, returns Lipschitz constant
        w.r.t. to states.

        If p!=1, p!=2 or p!=np.inf, returns an upper bound on the induced norm
        """
        if self.p == 1:
            order = np.inf
        else:
            order = self.p / (self.p - 1.0)

        if order in [1, 2]:
            return np.linalg.norm(self.A, ord=order)

        # If p!=1, p!=2 or p!=np.inf, return upper bound on the induced norm.
        return np.power(self.d, 1.0 / self.p) * np.linalg.norm(self.A,
                                                               ord=np.inf)


class PBall2D(RenderInterface2D, PBall):
    def __init__(self,
                 p=2,
                 action_list=[0.05 * np.array([1, 0]),
                              -0.05 * np.array([1, 0]),
                              0.05 * np.array([0, 1]),
                              -0.05 * np.array([0, 1])],
                 reward_amplitudes=np.array([1.0]),
                 reward_smoothness=np.array([0.25]),
                 reward_centers=[np.array([0.75, 0.0])],
                 A=np.eye(2),
                 B=np.eye(2),
                 sigma=0.01,
                 sigma_init=0.001,
                 mu_init=np.array([0.0, 0.0])
                 ):
        # Initialize PBall
        PBall.__init__(self, p, action_list, reward_amplitudes,
                       reward_smoothness,
                       reward_centers,
                       A, B, sigma, sigma_init, mu_init)

        # Render interface
        RenderInterface2D.__init__(self)

        # rendering info
        self.set_clipping_area((-1, 1, -1, 1))
        self.set_refresh_interval(50)  # in milliseconds

    def step(self, action):
        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(self.state.copy())
        return PBall.step(self, action)

    #
    # Code for rendering
    #

    def _get_ball_shape(self, xcenter, radius):
        shape = GeometricPrimitive("POLYGON")
        n_points = 200
        theta_vals = np.linspace(0.0, 2 * np.pi, n_points)
        for theta in theta_vals:
            pp = np.array([2.0 * np.cos(theta), 2.0 * np.sin(theta)])
            pp = xcenter + radius * projection_to_pball(pp, self.p)
            # project to the main ball after translation
            pp = projection_to_pball(pp, self.p)
            shape.add_vertex((pp[0], pp[1]))
        return shape

    def get_background(self):
        bg = Scene()

        # ball shape
        contour = self._get_ball_shape(np.zeros(2), 1.0)
        contour.set_color((0.0, 0.0, 0.5))
        bg.add_shape(contour)

        # reward position
        for ii, ampl in enumerate(self.reward_amplitudes):
            contour = self._get_ball_shape(self.reward_centers[ii],
                                           self.reward_smoothness[ii])
            ampl = 1.0 - ampl  # dark violet = more reward
            contour.set_color((0.5, 0.0, 0.5 * (1.0 + ampl)))
            bg.add_shape(contour)

        return bg

    def get_scene(self, state):
        scene = Scene()

        agent = GeometricPrimitive("QUADS")
        agent.set_color((0.75, 0.0, 0.5))
        size = 0.05
        x = state[0]
        y = state[1]
        agent.add_vertex((x - size / 4.0, y - size))
        agent.add_vertex((x + size / 4.0, y - size))
        agent.add_vertex((x + size / 4.0, y + size))
        agent.add_vertex((x - size / 4.0, y + size))

        agent.add_vertex((x - size, y - size / 4.0))
        agent.add_vertex((x + size, y - size / 4.0))
        agent.add_vertex((x + size, y + size / 4.0))
        agent.add_vertex((x - size, y + size / 4.0))

        scene.add_shape(agent)
        return scene


class SimplePBallND(PBall):
    """
    PBall environment in d dimensions with simple dynamics.
    """

    def __init__(self,
                 p=2,
                 dim=2,
                 action_amplitude=0.05,
                 r_smoothness=0.25,
                 sigma=0.01,
                 sigma_init=0.001,
                 mu_init=None
                 ):
        # Action list
        action_list = []
        for dd in range(dim):
            aux = np.zeros(dim)
            aux[dd] = action_amplitude
            action_list.append(aux)
            action_list.append(-1 * aux)

        # Rewards
        reward_amplitudes = np.array([1.0])
        reward_smoothness = np.array([r_smoothness])
        reward_centers = [np.zeros(dim)]
        reward_centers[0][0] = 0.8

        # Transitions
        A = np.eye(dim)
        B = np.eye(dim)

        # Initial position
        if mu_init is None:
            mu_init = np.zeros(dim)

        # Initialize PBall
        PBall.__init__(self, p, action_list, reward_amplitudes,
                       reward_smoothness,
                       reward_centers,
                       A, B, sigma, sigma_init, mu_init)

# if __name__ == '__main__':
#     env = PBall2D(p=5)
#     print(env.get_transitions_lipschitz_constant())
#     print(env.get_reward_lipschitz_constant())

#     env.enable_rendering()

#     for ii in range(100):
#         env.step(1)
#         env.step(3)

#     env.render()
