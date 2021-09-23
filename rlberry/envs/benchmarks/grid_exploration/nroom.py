import logging
import math
import numpy as np
import rlberry.spaces as spaces
from rlberry.envs.finite import GridWorld
from rlberry.rendering import Scene, GeometricPrimitive

logger = logging.getLogger(__name__)


def get_nroom_state_coord(state_index, nroom_env):
    yy, xx = nroom_env.index2coord[state_index]
    # centering
    xx = xx + 0.5
    yy = yy + 0.5
    # map to [0, 1]
    xx = xx / nroom_env.ncols
    yy = yy / nroom_env.nrows
    return np.array([xx, yy])


class NRoom(GridWorld):
    """
    GridWorld with N rooms of size L x L. The agent starts in the middle room.

    There is one small and easy reward in the first room,
    one big reward in the last room and zero reward elsewhere.

    There is a 5% error probability in the transitions when taking an action.

    Parameters
    ----------
    nrooms : int
        Number of rooms.
    reward_free : bool, default=False
        If true, no rewards are given to the agent.
    array_observation:
        If true, the observations are converted to an array (x, y)
        instead of a discrete index.
        The underlying discrete space is saved in env.discrete_observation_space.
    room_size : int
        Dimension (L) of each room (L x L).
    success_probability : double, default: 0.95
        Sucess probability of an action. A failure is going to the wrong direction.
    remove_walls : bool, default: False
        If True, remove walls. Useful for debug.
    initial_state_distribution: {'center', 'uniform'}
        If 'center', always start at the center.
        If 'uniform', start anywhere with uniform probability.
    include_traps: bool, default: False
        If true, each room will have a terminal state (a "trap").
    Notes
    -----
    The function env.sample() does not handle conversions to array states
    when array_observation is True. Only the functions env.reset() and
    env.step() are covered.
    """
    name = "N-Room"

    def __init__(self,
                 nrooms=7,
                 reward_free=False,
                 array_observation=False,
                 room_size=5,
                 success_probability=0.95,
                 remove_walls=False,
                 initial_state_distribution='center',
                 include_traps=False):

        assert nrooms > 0, "nrooms must be > 0"
        assert initial_state_distribution in ('center', 'uniform')

        self.reward_free = reward_free
        self.array_observation = array_observation
        self.nrooms = nrooms

        # Max number of rooms/columns per row
        self.max_rooms_per_row = 5

        # Room size (default = 5x5)
        self.room_size = room_size

        # Grid size
        self.room_nrows = math.ceil(nrooms / self.max_rooms_per_row)
        if self.room_nrows > 1:
            self.room_ncols = self.max_rooms_per_row
        else:
            self.room_ncols = nrooms
        nrows = self.room_size * self.room_nrows + (self.room_nrows - 1)
        ncols = self.room_size * self.room_ncols + (self.room_ncols - 1)

        # # walls
        walls = []
        for room_col in range(self.room_ncols - 1):
            col = (room_col + 1) * (self.room_size + 1) - 1
            for jj in range(nrows):
                if (jj % (self.room_size + 1)) != (self.room_size // 2):
                    walls.append((jj, col))

        for room_row in range(self.room_nrows - 1):
            row = (room_row + 1) * (self.room_size + 1) - 1
            for jj in range(ncols):
                walls.append((row, jj))

        # process each room
        start_coord = None
        terminal_state = None
        self.traps = []
        count = 0
        for room_r in range(self.room_nrows):
            if room_r % 2 == 0:
                cols_iterator = range(self.room_ncols)
            else:
                cols_iterator = reversed(range(self.room_ncols))
            for room_c in cols_iterator:
                # existing rooms
                if count < self.nrooms:
                    # remove top wall
                    if ((room_c == self.room_ncols - 1) and (room_r % 2 == 0)) \
                            or ((room_c == 0) and (room_r % 2 == 1)):
                        if room_r != self.room_nrows - 1:
                            wall_to_remove = self._convert_room_coord_to_global(
                                room_r, room_c,
                                self.room_size, self.room_size // 2)
                            if wall_to_remove in walls:
                                walls.remove(wall_to_remove)
                # rooms to remove
                else:
                    for ii in range(-1, self.room_size + 1):
                        for jj in range(-1, self.room_size + 1):
                            wall_to_include = self._convert_room_coord_to_global(
                                room_r, room_c,
                                ii, jj)
                            if wall_to_include[0] >= 0 and wall_to_include[0] < nrows \
                                    and wall_to_include[1] >= 0 and wall_to_include[1] < ncols \
                                    and (wall_to_include not in walls):
                                walls.append(wall_to_include)
                    pass

                # start coord
                if count == nrooms // 2:
                    start_coord = self._convert_room_coord_to_global(
                        room_r, room_c,
                        self.room_size // 2, self.room_size // 2)
                # terminal state
                if count == nrooms - 1:
                    terminal_state = self._convert_room_coord_to_global(
                        room_r, room_c,
                        self.room_size // 2, self.room_size // 2)
                # trap
                if include_traps:
                    self.traps.append(
                        self._convert_room_coord_to_global(
                            room_r, room_c,
                            self.room_size // 2 + 1, self.room_size // 2 + 1)
                    )
                count += 1

        terminal_states = (terminal_state,) + tuple(self.traps)

        if self.reward_free:
            reward_at = {}
        else:
            reward_at = {
                terminal_state: 1.0,
                start_coord: 0.01,
                (self.room_size // 2, self.room_size // 2): 0.1
            }

        # Check remove_walls
        if remove_walls:
            walls = ()

        # Init base class
        GridWorld.__init__(self,
                           nrows=nrows,
                           ncols=ncols,
                           start_coord=start_coord,
                           terminal_states=terminal_states,
                           success_probability=success_probability,
                           reward_at=reward_at,
                           walls=walls,
                           default_reward=0.0)

        # Check initial distribution
        if initial_state_distribution == 'uniform':
            distr = np.ones(self.observation_space.n) / self.observation_space.n
            self.set_initial_state_distribution(distr)

        # spaces
        if self.array_observation:
            self.discrete_observation_space = self.observation_space
            self.observation_space = spaces.Box(0.0, 1.0, shape=(2,))

    def _convert_room_coord_to_global(self, room_row, room_col, room_coord_row, room_coord_col):
        col_offset = (self.room_size + 1) * room_col
        row_offset = (self.room_size + 1) * room_row

        row = room_coord_row + row_offset
        col = room_coord_col + col_offset
        return (row, col)

    def _convert_index_to_float_coord(self, state_index):
        yy, xx = self.index2coord[state_index]

        # centering
        xx = xx + 0.5
        yy = yy + 0.5
        # map to [0, 1]
        xx = xx / self.ncols
        yy = yy / self.nrows
        return np.array([xx, yy])

    def reset(self):
        self.state = GridWorld.reset(self)
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

        # traps
        for (y, x) in self.traps:
            shape = GeometricPrimitive("POLYGON")
            shape.set_color((0.5, 0.0, 0.0))
            shape.add_vertex((x, y))
            shape.add_vertex((x + 1, y))
            shape.add_vertex((x + 1, y + 1))
            shape.add_vertex((x, y + 1))
            bg.add_shape(shape)

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
            flag = GeometricPrimitive("POLYGON")
            rwd = self.reward_at[(y, x)]
            if rwd == 1.0:
                flag.set_color((0.0, 0.5, 0.0))
            elif rwd == 0.1:
                flag.set_color((0.0, 0.0, 0.5))
            else:
                flag.set_color((0.5, 0.0, 0.0))

            x += 0.5
            y += 0.25
            flag.add_vertex((x, y))
            flag.add_vertex((x + 0.25, y + 0.5))
            flag.add_vertex((x - 0.25, y + 0.5))
            bg.add_shape(flag)

        return bg
