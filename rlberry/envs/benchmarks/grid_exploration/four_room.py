import logging
import numpy as np
import rlberry.spaces as spaces
from rlberry.envs.finite import GridWorld

logger = logging.getLogger(__name__)


class FourRoom(GridWorld):
    """
    GridWorld with four rooms.

    Parameters
    ----------
    reward_free : bool, default=False
        If true, no rewards are given to the agent.
    difficulty: int, {0, 1 or 2}
        Difficulty 0: reward in one location
        Difficulty 1: easy suboptimal reward, hard optimal reward
        Difficulty 2: easy suboptimal reward, hard optimal reward,
            negative rewards by default.
        Note: this parameter is ignored if reward_free is True.
    array_observation:
        If true, the observations are converted to an array (x, y)
        instead of a discrete index.

    Notes
    -----
    The function env.sample() does not handle conversions to array states
    when array_observation is True. Only the functions env.reset() and
    env.step() are covered.
    """
    name = "FourRoom"

    def __init__(self,
                 reward_free=False,
                 difficulty=0,
                 array_observation=False):
        self.reward_free = reward_free
        self.difficulty = difficulty
        self.array_observation = array_observation

        if difficulty not in [0, 1, 2]:
            raise ValueError("FourRoom difficulty must be in [0, 1, 2]")

        # Common parameters
        nrows = 9
        ncols = 9
        start_coord = (0, 0)
        terminal_states = ((8, 0),)
        success_probability = 0.95
        #
        walls = ()
        for ii in range(9):
            if ii not in [2, 6]:
                walls += ((ii, 4),)
        for jj in range(9):
            if jj != 7:
                walls += ((4, jj),)

        # Default reward according to the difficulty
        if difficulty in [0, 1]:
            default_reward = 0.0
        elif difficulty == 2:
            default_reward = -0.005

        # Rewards according to the difficulty
        if self.reward_free:
            reward_at = {}
        else:
            if difficulty == 0:
                reward_at = {(8, 0): 1.0}
            elif difficulty in [1, 2]:
                reward_at = {
                    (8, 0): 1.0,
                    (3, 3): 0.1,
                }

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
        xx = xx / self.ncols
        yy = yy / self.nrows
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
