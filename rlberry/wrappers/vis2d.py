from rlberry.envs import Wrapper
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from rlberry.rendering.utils import video_write
import gym.spaces as spaces
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


logger = logging.getLogger(__name__)


class Transition:
    def __init__(self, state, action, reward, n_total_visits, n_episode_visits):
        self.state = state
        self.action = action
        self.reward = reward
        self.n_total_visits = n_total_visits
        self.n_episode_visits = n_episode_visits


class TrajectoryMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.clear()

    def clear(self):
        self.n_trajectories = 0
        # current trajectory
        self.current_traj_transitions = []
        # all trajectories
        self.trajectories = []

    def append(self, transition):
        self.current_traj_transitions.append(transition)

    def end_trajectory(self):
        if len(self.current_traj_transitions) == 0:
            return

        self.n_trajectories += 1

        # store data
        self.trajectories.append(self.current_traj_transitions)

        if self.n_trajectories > self.max_size:
            self.trajectories.pop(0)
            self.n_trajectories = self.max_size

        # reset data
        self.current_traj_transitions = []

    def is_empty(self):
        return self.n_trajectories > 0


class Vis2dWrapper(Wrapper):
    """
    Stores and visualizes the trajectories environments with 2d box observation spaces
    and discrete action spaces.

    Parameters
    ----------
    env: gym.Env
    n_bins_obs : int, default = 10
        Number of intervals to discretize each dimension of the observation space.
        Used to count number of visits.
    memory_size : int, default = 100
        Maximum number of trajectories to keep in memory.
        The most recent ones are kept.
    """
    def __init__(self,
                 env,
                 n_bins_obs=10,
                 memory_size=100):
        Wrapper.__init__(self, env)
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Discrete)

        self.memory = TrajectoryMemory(memory_size)
        self.total_visit_counter = DiscreteCounter(self.env.observation_space,
                                                   self.env.action_space,
                                                   n_bins_obs=n_bins_obs)
        self.episode_visit_counter = DiscreteCounter(self.env.observation_space,
                                                     self.env.action_space,
                                                     n_bins_obs=n_bins_obs)
        self.current_state = None
        self.curret_step = 0

    def reset(self):
        self.current_step = 0
        self.current_state = self.env.reset()
        return self.current_state

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # initialize new trajectory
        if self.current_step == 0:
            self.memory.end_trajectory()
            self.episode_visit_counter.reset()
        self.current_step += 1
        # update counters
        ss, aa = self.current_state, action
        ns = observation
        self.total_visit_counter.update(ss, aa, ns, reward)
        self.episode_visit_counter.update(ss, aa, ns, reward)
        # store transition
        transition = Transition(ss, aa, reward,
                                self.total_visit_counter.count(ss, aa),
                                self.episode_visit_counter.count(ss, aa))
        self.memory.append(transition)
        # update current state
        self.current_state = observation
        return observation, reward, done, info

    def plot(self,
             fignum=None,
             figsize=(6, 6),
             hide_axis=True,
             show=True,
             video_filename=None,
             colormap_fn=None,
             framerate=15,
             n_skip=1,
             dot_scale_factor=2.5):
        """
        If video_filename is given, a video file is saved. Otherwise,
        plot only the final frame.

        Parameters
        ----------
        fignum : str
            Figure name
        figsize : (float, float)
            (width, height) of the image in inches.
        hide_axis : bool
            If True, axes are hidden.
        show : bool
            If True, calls plt.show()
        video_filename : str or None
            If not None, save a video with given filename.
        colormap_fn : callable
            Colormap function, e.g. matplotlib.cm.cool (default)
        framerate : int, default: 15
            Video framerate.
        n_skip : int, default: 1
            Skip period: every n_skip trajectories, one trajectory is plotted.
        dot_scale_factor : double
            Scale factor for scatter plot points.
        """
        logger.info("Plotting...")

        fignum = fignum or str(self)
        colormap_fn = colormap_fn or cm.cool

        # discretizer
        discretizer = self.episode_visit_counter.state_discretizer
        epsilon = min(discretizer._bins[0][1] - discretizer._bins[0][0],
                      discretizer._bins[1][1] - discretizer._bins[1][0])

        # figure setup
        xlim = [self.env.observation_space.low[0],  self.env.observation_space.high[0]]
        ylim = [self.env.observation_space.low[1],  self.env.observation_space.high[1]]

        fig = plt.figure(fignum, figsize=figsize)
        fig.clf()
        canvas = FigureCanvas(fig)
        images = []
        ax = fig.gca()

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if hide_axis:
            ax.set_axis_off()

        # scatter plot
        indices = np.arange(self.memory.n_trajectories)[::n_skip]

        for idx in indices:
            traj = self.memory.trajectories[idx]
            color_time_intensity = (idx+1)/self.memory.n_trajectories
            color = colormap_fn(color_time_intensity)

            states = np.array([traj[ii].state for ii in range(len(traj))])

            sizes = np.array(
                [traj[ii].n_episode_visits for ii in range(len(traj))]
            )

            sizes = 1 + sizes
            sizes = (dot_scale_factor**2) * 100 * epsilon * sizes / sizes.max()

            ax.scatter(x=states[:, 0], y=states[:, 1], color=color, s=sizes)
            plt.tight_layout()

            if video_filename is not None:
                canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(image_from_plot)

        if video_filename is not None:
            logger.info("... writing video ...")
            video_write(video_filename, images, framerate=framerate)

        logger.info("... done!")

        if show:
            plt.show()
