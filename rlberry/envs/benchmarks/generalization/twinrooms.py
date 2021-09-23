import logging
import numpy as np
import rlberry.spaces as spaces
from rlberry.envs import Model
from rlberry.rendering import Scene, GeometricPrimitive, RenderInterface2D
from rlberry.rendering.common_shapes import circle_shape

logger = logging.getLogger(__name__)


class TwinRooms(RenderInterface2D, Model):
    """
    Two continuous grid worlds, side by side, separated by a wall.
    Both are identical (or almost identical), and the agent has equal probability to
    start in any of the two rooms.

    It can be used to test the generalization capability of agents:
    a policy learned in one of the rooms can be used to learn faster
    a policy in the other room.

    There are 4 actions, one for each direction (left/right/up/down).

    Parameters
    ----------
    noise_room1: double, default: 0.01
        Noise in the transitions of the first room.
    noise_room2: double, default: 0.01
        Noise in the transitions of the second room.

    Notes
    -----
    The function env.sample() does not handle conversions to array states
    when array_observation is True. Only the functions env.reset() and
    env.step() are covered.
    """
    name = "TwinRooms"

    def __init__(self,
                 noise_room1=0.01,
                 noise_room2=0.01):
        Model.__init__(self)
        RenderInterface2D.__init__(self)

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([2.0, 1.0]),
        )
        self.action_space = spaces.Discrete(4)
        self.reward_range = (0.0, 1.0)

        self.room_noises = [noise_room1, noise_room2]

        # environment parameters
        self.action_displacement = 0.1
        self.wall_eps = 0.05

        # base reward position
        self.base_reward_pos = np.array([0.8, 0.8])

        # rendering info
        self.set_clipping_area((0, 2, 0, 1))
        self.set_refresh_interval(100)  # in milliseconds
        self.renderer_type = 'opengl'

        # reset
        self.reset()

    def reset(self):
        self.current_room = self.seeder.rng.integers(2)
        if self.current_room == 0:
            self.state = np.array([0.1, 0.1])
        else:
            self.state = np.array([1.1, 0.1])
        return self.state.copy()

    def _reward_fn(self, state):
        # max reward at (x, y) = reward_pos
        reward_pos = self.base_reward_pos
        if self.current_room == 1:
            reward_pos = reward_pos + np.array([1.0, 0.0])
        xr, yr = reward_pos

        dist = np.sqrt((state[0] - xr) ** 2.0 + (state[1] - yr) ** 2.0)
        reward = max(0.0, 1.0 - dist / 0.1)
        return reward

    def _clip_to_room(self, state):
        state[1] = max(0.0, state[1])
        state[1] = min(1.0, state[1])
        if self.current_room == 0:
            state[0] = max(0.0, state[0])
            state[0] = min(1.0 - self.wall_eps, state[0])
        else:
            state[0] = max(1.0 + self.wall_eps, state[0])
            state[0] = min(2.0, state[0])
        return state

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(self.state)

        next_state, reward, done, info = self.sample(self.state, action)
        self.state = next_state
        return self.state.copy(), reward, done, info

    def sample(self, state, action):
        delta = self.action_displacement
        if action == 0:
            displacement = np.array([delta, 0.0])
        elif action == 1:
            displacement = np.array([-delta, 0.0])
        elif action == 2:
            displacement = np.array([0.0, delta])
        elif action == 3:
            displacement = np.array([0.0, -delta])
        else:
            raise ValueError("Invalid action")

        next_state = state + displacement \
                     + self.room_noises[self.current_room] * self.rng.normal(size=2)

        # clip to room
        next_state = self._clip_to_room(next_state)

        reward = self._reward_fn(state)
        done = False
        info = {}

        return next_state, reward, done, info

    #
    # Code for rendering
    #

    def get_background(self):
        """
        Returne a scene (list of shapes) representing the background
        """
        bg = Scene()

        # wall
        eps = self.wall_eps
        shape = GeometricPrimitive("POLYGON")
        shape.set_color((0.25, 0.25, 0.25))
        shape.add_vertex((1 - eps, 0))
        shape.add_vertex((1 - eps, 1))
        shape.add_vertex((1 + eps, 1))
        shape.add_vertex((1 + eps, 0))
        bg.add_shape(shape)

        # rewards
        for (x, y) in [self.base_reward_pos, self.base_reward_pos + np.array([1.0, 0.0])]:
            reward = circle_shape((x, y), 0.1, n_points=50)
            reward.type = "POLYGON"
            reward.set_color((0.0, 0.5, 0.0))
            bg.add_shape(reward)

        return bg

    def get_scene(self, state):
        """
        Return scene (list of shapes) representing a given state
        """
        x, y = state
        scene = Scene()
        agent = circle_shape((x, y), 0.02, n_points=5)
        agent.type = "POLYGON"
        agent.set_color((0.75, 0.0, 0.5))
        scene.add_shape(agent)
        return scene
