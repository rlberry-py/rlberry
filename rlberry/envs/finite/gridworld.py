import matplotlib
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib import cm

from rlberry.envs.finite import FiniteMDP
from rlberry.envs.finite import gridworld_utils
from rlberry.rendering import Scene, GeometricPrimitive, RenderInterface2D
from rlberry.rendering.common_shapes import circle_shape


logger = logging.getLogger(__name__)


class GridWorld(RenderInterface2D, FiniteMDP):
    """
    Simple GridWorld environment.

    Parameters
    -----------
    nrows : int
        number of rows
    ncols : int
        number of columns
    start_coord : tuple
        tuple with coordinates of initial position
    terminal_states : tuple
        ((row_0, col_0), (row_1, col_1), ...) = coordinates of
        terminal states
    success_probability : double
        probability of moving in the chosen direction
    reward_at: dict
        dictionary, keys = tuple containing coordinates, values = reward
        at each coordinate
    walls : tuple
        ((row_0, col_0), (row_1, col_1), ...) = coordinates of walls
    default_reward : double
        reward received at states not in  'reward_at'

    """

    name = "GridWorld"

    def __init__(
        self,
        nrows=5,
        ncols=5,
        start_coord=(0, 0),
        terminal_states=None,
        success_probability=0.9,
        reward_at=None,
        walls=((1, 1), (2, 2)),
        default_reward=0.0,
    ):
        # Grid dimensions
        self.nrows = nrows
        self.ncols = ncols

        # Reward parameters
        self.default_reward = default_reward

        # Default config
        if reward_at is not None:
            self.reward_at = reward_at
        else:
            self.reward_at = {(nrows - 1, ncols - 1): 1}
        if walls is not None:
            self.walls = walls
        else:
            self.walls = ()
        if terminal_states is not None:
            self.terminal_states = terminal_states
        else:
            self.terminal_states = ()

        # Probability of going left/right/up/down when choosing the
        # correspondent action
        # The remaining probability mass is distributed uniformly to other
        # available actions
        self.success_probability = success_probability

        # Start coordinate
        self.start_coord = tuple(start_coord)

        # Actions (string to index & index to string)
        self.a_str2idx = {"left": 0, "right": 1, "down": 2, "up": 3}
        self.a_idx2str = {0: "left", 1: "right", 2: "down", 3: "up"}

        # --------------------------------------------
        # The variables below are defined in _build()
        # --------------------------------------------

        # Mappings (state index) <-> (state coordinate)
        self.index2coord = {}
        self.coord2index = {}

        # MDP parameters for base class
        self.P = None
        self.R = None
        self.Ns = None
        self.Na = 4

        # Build
        self._build()
        init_state_idx = self.coord2index[start_coord]
        FiniteMDP.__init__(
            self, self.R, self.P, initial_state_distribution=init_state_idx
        )
        RenderInterface2D.__init__(self)
        self.reset()
        self.reward_range = (self.R.min(), self.R.max())

        # rendering info
        self.set_clipping_area((0, self.ncols, 0, self.nrows))
        self.set_refresh_interval(100)  # in milliseconds
        self.renderer_type = "pygame"

    @classmethod
    def from_layout(
        cls, layout: str = gridworld_utils.DEFAULT_LAYOUT, success_probability=0.95
    ):
        """
        Create GridWorld instance from a layout.

        Layout symbols:

        '#' : wall
        'r' : reward of 1, terminal state
        'R' : reward of 1, non-terminal state
        'T' : terminal state
        'I' : initial state (if several, start uniformly among I)
        'O' : empty state
        any other char : empty state

        Layout example:

        IOOOO # OOOOO  O OOOOR
        OOOOO # OOOOO  # OOOOO
        OOOOO O OOOOO  # OOOOO
        OOOOO # OOOOO  # OOOOO
        IOOOO # OOOOO  # OOOOr
        """
        info = gridworld_utils.get_layout_info(layout)
        nrows = info["nrows"]
        ncols = info["ncols"]
        walls = info["walls"]
        reward_at = info["reward_at"]
        terminal_states = info["terminal_states"]
        initial_states_coord = info["initial_states"]

        # Init base class
        env = cls(
            nrows=nrows,
            ncols=ncols,
            terminal_states=terminal_states,
            success_probability=success_probability,
            reward_at=reward_at,
            walls=walls,
            default_reward=0.0,
        )

        # Set initial distribution
        distr = np.zeros(env.observation_space.n)
        for init_coord in initial_states_coord:
            init_index = env.coord2index[init_coord]
            distr[init_index] = 1.0
        distr = distr / distr.sum()
        env.set_initial_state_distribution(distr)

        return env

    def is_terminal(self, state):
        state_coord = self.index2coord[state]
        return state_coord in self.terminal_states

    def reward_fn(self, state, action, next_state):
        row, col = self.index2coord[state]
        if (row, col) in self.reward_at:
            return self.reward_at[(row, col)]
        if (row, col) in self.walls:
            return 0.0
        return self.default_reward

    def _build(self):
        self._build_state_mappings_and_states()
        self._build_transition_probabilities()
        self._build_mean_rewards()

    def _build_state_mappings_and_states(self):
        index = 0
        for rr in range(self.nrows):
            for cc in range(self.ncols):
                if (rr, cc) in self.walls:
                    self.coord2index[(rr, cc)] = -1
                else:
                    self.coord2index[(rr, cc)] = index
                    self.index2coord[index] = (rr, cc)
                    index += 1
        states = np.arange(index).tolist()
        self.Ns = len(states)

    def _build_mean_rewards(self):
        S = self.Ns
        A = self.Na
        self.R = np.zeros((S, A))
        for ss in range(S):
            for aa in range(A):
                mean_r = 0
                for ns in range(S):
                    mean_r += self.reward_fn(ss, aa, ns) * self.P[ss, aa, ns]
                self.R[ss, aa] = mean_r

    def _build_transition_probabilities(self):
        Ns = self.Ns
        Na = self.Na
        self.P = np.zeros((Ns, Na, Ns))
        for s in range(Ns):
            s_coord = self.index2coord[s]
            neighbors = self._get_neighbors(*s_coord)
            valid_neighbors = [neighbors[nn][0] for nn in neighbors if neighbors[nn][1]]
            n_valid = len(valid_neighbors)
            for a in range(Na):  # each action corresponds to a direction
                for nn in neighbors:
                    next_s_coord = neighbors[nn][0]
                    if next_s_coord in valid_neighbors:
                        next_s = self.coord2index[next_s_coord]
                        if a == nn:  # action is successful
                            self.P[s, a, next_s] = self.success_probability + (
                                1 - self.success_probability
                            ) * (n_valid == 1)
                        elif neighbors[a][0] not in valid_neighbors:
                            self.P[s, a, s] = 1.0
                        else:
                            if n_valid > 1:
                                self.P[s, a, next_s] = (
                                    1.0 - self.success_probability
                                ) / (n_valid - 1)

    def _get_neighbors(self, row, col):
        aux = {}
        aux["left"] = (row, col - 1)  # left
        aux["right"] = (row, col + 1)  # right
        aux["up"] = (row - 1, col)  # up
        aux["down"] = (row + 1, col)  # down
        neighbors = {}
        for direction_str in aux:
            direction = self.a_str2idx[direction_str]
            next_s = aux[direction_str]
            neighbors[direction] = (next_s, self._is_valid(*next_s))
        return neighbors

    def get_transition_support(self, state):
        row, col = self.index2coord[state]
        neighbors = [(row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col)]
        return [
            self.coord2index[coord] for coord in neighbors if self._is_valid(*coord)
        ]

    def _is_valid(self, row, col):
        if (row, col) in self.walls:
            return False
        elif row < 0 or row >= self.nrows:
            return False
        elif col < 0 or col >= self.ncols:
            return False
        return True

    def _build_ascii(self):
        grid = [[""] * self.ncols for rr in range(self.nrows)]
        grid_idx = [[""] * self.ncols for rr in range(self.nrows)]
        for rr in range(self.nrows):
            for cc in range(self.ncols):
                if (rr, cc) in self.walls:
                    grid[rr][cc] = "x "
                else:
                    grid[rr][cc] = "o "
                grid_idx[rr][cc] = str(self.coord2index[(rr, cc)]).zfill(3)

        for (rr, cc) in self.reward_at:
            rwd = self.reward_at[(rr, cc)]
            if rwd > 0:
                grid[rr][cc] = "+ "
            if rwd < 0:
                grid[rr][cc] = "-"

        grid[self.start_coord[0]][self.start_coord[1]] = "I "

        # current position of the agent
        x, y = self.index2coord[self.state]
        grid[x][y] = "A "

        #
        grid_ascii = ""
        for rr in range(self.nrows + 1):
            if rr < self.nrows:
                grid_ascii += str(rr).zfill(2) + 2 * " " + " ".join(grid[rr]) + "\n"
            else:
                grid_ascii += 3 * " " + " ".join(
                    [str(jj).zfill(2) for jj in range(self.ncols)]
                )

        self.grid_ascii = grid_ascii
        self.grid_idx = grid_idx
        return self.grid_ascii

    def display_values(self, values):
        assert len(values) == self.Ns
        grid_values = [["X".ljust(9)] * self.ncols for ii in range(self.nrows)]
        for s_idx in range(self.Ns):
            v = values[s_idx]
            row, col = self.index2coord[s_idx]
            grid_values[row][col] = ("%0.2f" % v).ljust(9)

        grid_values_ascii = ""
        for rr in range(self.nrows + 1):
            if rr < self.nrows:
                grid_values_ascii += (
                    str(rr).zfill(2) + 2 * " " + " ".join(grid_values[rr]) + "\n"
                )
            else:
                grid_values_ascii += 4 * " " + " ".join(
                    [str(jj).zfill(2).ljust(9) for jj in range(self.ncols)]
                )
        logger.info(grid_values_ascii)

    def print_transition_at(self, row, col, action):
        s_idx = self.coord2index[(row, col)]
        if s_idx < 0:
            logger.info("wall!")
            return
        a_idx = self.a_str2idx[action]
        for next_s_idx, prob in enumerate(self.P[s_idx, a_idx]):
            if prob > 0:
                logger.info(
                    "to (%d, %d) with prob %f"
                    % (self.index2coord[next_s_idx] + (prob,))
                )

    def render_ascii(self):
        print(self._build_ascii())

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(self.state)

        # take step
        next_state, reward, done, info = self.sample(self.state, action)
        self.state = next_state
        return next_state, reward, done, info

    #
    # Code for rendering
    #
    def get_layout_array(self, state_data=None, fill_walls_with=np.nan):
        """
        Returns an array 'layout' of shape (nrows, ncols) such that:

            layout[row, col] = state_data[self.coord2idx[row, col]]

        If (row, col) is a wall:

            layout[row, col] = fill_walls_with

        Parameters
        ----------
        state_data : np.array, default = None
            Array of shape (self.observation_space.n,)
        fill_walls_with : float, default: np.nan
            Value to set in the layout in the coordinates corresponding to walls.

        Returns
        -------
        Gridworld layout array of shape (nrows, ncols).
        """
        layout = np.zeros((self.nrows, self.ncols))
        if state_data is not None:
            assert state_data.shape == (self.observation_space.n,)
            data_rows = [self.index2coord[idx][0] for idx in self.index2coord]
            data_cols = [self.index2coord[idx][1] for idx in self.index2coord]
            layout[data_rows, data_cols] = state_data
        else:
            state_rr, state_cc = self.index2coord[self.state]
            layout[state_rr, state_cc] = 1.0

        walls_rows = [ww[0] for ww in self.walls]
        walls_cols = [ww[1] for ww in self.walls]
        layout[walls_rows, walls_cols] = fill_walls_with
        return layout

    def get_layout_img(
        self, state_data=None, colormap_name="cool", wall_color=(0.0, 0.0, 0.0)
    ):
        """
        Returns an image array representing the value of `state_data` on
        the gridworld layout.

        Parameters
        ----------
        state_data : np.array, default = None
            Array of shape (self.observation_space.n,)
        colormap_name : str, default = 'cool'
            Colormap name.
            See https://matplotlib.org/tutorials/colors/colormaps.html
        wall_color : tuple
            RGB color for walls.
        Returns
        -------
        Gridworld image array of shape (nrows, ncols, 3).
        """
        # map data to [0.0, 1.0]
        if state_data is not None:
            state_data = state_data - state_data.min()
            if state_data.max() > 0.0:
                state_data = state_data / state_data.max()

        colormap_fn = plt.get_cmap(colormap_name)
        layout = self.get_layout_array(state_data, fill_walls_with=np.nan)
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap_fn)
        img = np.zeros(layout.shape + (3,))
        for rr in range(layout.shape[0]):
            for cc in range(layout.shape[1]):
                if np.isnan(layout[rr, cc]):
                    img[self.nrows - 1 - rr, cc, :] = wall_color
                else:
                    img[self.nrows - 1 - rr, cc, :3] = scalar_map.to_rgba(
                        layout[rr, cc]
                    )[:3]
        return img

    def get_background(self):
        """
        Return a scene (list of shapes) representing the background
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
            flag = GeometricPrimitive("POLYGON")
            rwd = self.reward_at[(y, x)]
            color = 0.5 * np.abs(rwd) / self.reward_range[1]
            if rwd > 0:
                flag.set_color((0.0, color, 0.0))
            if rwd < 0:
                flag.set_color((color, 0.0, 0.0))

            x += 0.5
            y += 0.25
            flag.add_vertex((x, y))
            flag.add_vertex((x + 0.25, y + 0.5))
            flag.add_vertex((x - 0.25, y + 0.5))
            bg.add_shape(flag)

        return bg

    def get_scene(self, state):
        """
        Return scene (list of shapes) representing a given state
        """
        y, x = self.index2coord[state]
        x = x + 0.5  # centering
        y = y + 0.5  # centering

        scene = Scene()

        agent = circle_shape((x, y), 0.25, n_points=5)
        agent.type = "POLYGON"
        agent.set_color((0.75, 0.0, 0.5))

        scene.add_shape(agent)
        return scene
