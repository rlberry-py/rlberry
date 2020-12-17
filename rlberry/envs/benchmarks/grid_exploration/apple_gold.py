import logging
import numpy as np
import rlberry.spaces as spaces
from rlberry.envs.finite import GridWorld
from rlberry.rendering import Scene, GeometricPrimitive

logger = logging.getLogger(__name__)


class AppleGold(GridWorld):
    """
    AppleGold with six rooms: this is merely a slightly modified
    version of SixRoom.

    Parameters
    ----------
    reward_free : bool, default=False
        If true, no rewards are given to the agent.
    array_observation:
        If true, the observations are converted to an array (x, y)
        instead of a discrete index.

    Notes
    -----
    The function env.sample() does not handle conversions to array states
    when array_observation is True. Only the functions env.reset() and
    env.step() are covered.

    Reference
    ---------
    .. seaalso::
        Guo et al.: Self-Imitation Learning via
        Trajectory-Conditioned Policy
        for Hard-Exploration Tasks
        arXiv preprint arXiv:1907.10247
    """
    name = "AppleGold"

    def __init__(self, reward_free=False, array_observation=False):
        self.reward_free = reward_free
        self.array_observation = array_observation

        # Common parameters
        nrows = 13
        ncols = 17
        start_coord = (5, 1)
        terminal_states = ((7, 7),)
        success_probability = 0.95
        #
        walls = ()
        for ii in range(13):
            walls += ((ii, 0),)
            walls += ((ii, 16),)
        for jj in range(17):
            walls += ((0, jj),)
            walls += ((12, jj),)
        for ii in range(13):
            if ii not in [1, 11]:
                walls += ((ii, 6),)
                walls += ((ii, 10),)
        walls += ((11, 6),)
        for jj in range(17):
            if jj not in [1, 15]:
                walls += ((6, jj),)

        # Default reward according to the difficulty
        default_reward = 0

        # Rewards according to the difficulty
        if self.reward_free:
            reward_at = {}
        else:
            reward_at = {
                        (7, 7): 10.0,
                        (8, 2): 1.0,
                        (10, 3): 1.0
                        }
            for jj in range(7, 16):
                for ii in range(1, 12):
                    if (ii, jj) not in walls and (ii, jj) != (7, 7):
                        reward_at[(ii, jj)] = -0.05

        # Init base class
        GridWorld.__init__(self,
                           nrows=nrows,
                           ncols=ncols,
                           start_coord=start_coord,
                           terminal_states=terminal_states,
                           success_probability=success_probability,
                           reward_at=reward_at,
                           walls=walls,
                           default_reward=default_reward)

        # spaces
        if self.array_observation:
            self.observation_space = spaces.Box(0.0, 1.0, shape=(2,))

    def _convert_index_to_float_coord(self, state_index):
        yy, xx = self.index2coord[state_index]

        # centering
        xx = xx + 0.5
        yy = yy + 0.5
        # map to [0, 1]
        xx = xx/self.ncols
        yy = yy/self.nrows
        return np.array([xx, yy])

    def reset(self):
        self.state = self.coord2index[self.start_coord]
        state_to_return = self.state
        if self.array_observation:
            state_to_return = self._convert_index_to_float_coord(self.state)
        return state_to_return

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(self.state)

        # take step
        next_state, reward, done, info = self.sample(self.state, action)
        self.state = next_state

        state_to_return = self.state
        if self.array_observation:
            state_to_return = self._convert_index_to_float_coord(self.state)

        return state_to_return, reward, done, info

    def get_background(self):
        """
        Returne a scene (list of shapes) representing the background
        """
        bg = Scene()

        # walls
        for wall in self.walls:
            y, x = wall
            shape = GeometricPrimitive("POLYGON")
            shape.set_color((0.25, 0.25, 0.25))
            shape.add_vertex((x, y))
            shape.add_vertex((x + 1, y))
            shape.add_vertex((x + 1, y + 1))
            shape.add_vertex((x, y + 1))
            bg.add_shape(shape)

        # rewards
        for (y, x) in self.reward_at:
            rwd = self.reward_at[(y, x)]
            if rwd == -0.05:
                rock = GeometricPrimitive("POLYGON")
                rock.set_color((0.6, 0.6, 0.6))
                rock.add_vertex((x, y))
                rock.add_vertex((x + 1, y))
                rock.add_vertex((x + 1, y + 1))
                rock.add_vertex((x, y + 1))
                bg.add_shape(rock)
            else:
                flag = GeometricPrimitive("POLYGON")
                if rwd == 10:
                    flag.set_color((0.0, 0.5, 0.0))
                elif rwd == 1:
                    flag.set_color((0.0, 0.0, 0.5))

                x += 0.5
                y += 0.25
                flag.add_vertex((x, y))
                flag.add_vertex((x + 0.25, y + 0.5))
                flag.add_vertex((x - 0.25, y + 0.5))
                bg.add_shape(flag)

        return bg

