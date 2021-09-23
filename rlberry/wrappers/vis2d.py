from rlberry.envs import Wrapper
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from rlberry.rendering.utils import video_write
import gym.spaces as spaces
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class Transition:
    def __init__(self, raw_state, state, action, reward, n_total_visits, n_episode_visits):
        self.raw_state = raw_state
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


def identity(state, env, **kwargs):
    return state


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
    state_preprocess_fn : callable(state, env, **kwargs)-> np.ndarray, default: None
        Function that converts the state to a 2d array
    state_preprocess_kwargs : dict, default: None
        kwargs for state_preprocess_fn
    """

    def __init__(self,
                 env,
                 n_bins_obs=10,
                 memory_size=100,
                 state_preprocess_fn=None,
                 state_preprocess_kwargs=None):
        Wrapper.__init__(self, env)

        if state_preprocess_fn is None:
            assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Discrete)

        self.state_preprocess_fn = state_preprocess_fn or identity
        self.state_preprocess_kwargs = state_preprocess_kwargs or {}

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
        transition = Transition(ss,
                                self.state_preprocess_fn(ss, self.env, **self.state_preprocess_kwargs),
                                aa,
                                reward,
                                self.total_visit_counter.count(ss, aa),
                                self.episode_visit_counter.count(ss, aa))
        self.memory.append(transition)
        # update current state
        self.current_state = observation
        return observation, reward, done, info

    def plot_trajectories(self,
                          fignum=None,
                          figsize=(6, 6),
                          hide_axis=True,
                          show=True,
                          video_filename=None,
                          colormap_name='cool',
                          framerate=15,
                          n_skip=1,
                          dot_scale_factor=2.5,
                          alpha=0.25,
                          xlim=None,
                          ylim=None,
                          dot_size_means='episode_visits'):
        """
        Plot history of trajectories in a scatter plot.
        Colors distinguish recent and old trajectories, the size of the dots represent
        the number of visits to a state.

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
        colormap_name : str, default = 'cool'
            Colormap name.
            See https://matplotlib.org/tutorials/colors/colormaps.html
        framerate : int, default: 15
            Video framerate.
        n_skip : int, default: 1
            Skip period: every n_skip trajectories, one trajectory is plotted.
        dot_scale_factor : double
            Scale factor for scatter plot points.
        alpha : float, default: 0.25
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        xlim: list, default: None
            x plot limits, set to [0, 1] if None
        ylim: list, default: None
            y plot limits, set to [0, 1] if None
        dot_size_means : str, {'episode_visits' or 'total_visits'}, default: 'episode_visits'
            Whether to scale the dot size with the number of visits in an episode
            or the total number of visits during the whole interaction.
        """
        logger.info("Plotting...")

        fignum = fignum or str(self)
        colormap_fn = plt.get_cmap(colormap_name)

        # discretizer
        try:
            discretizer = self.episode_visit_counter.state_discretizer
            epsilon = min(discretizer._bins[0][1] - discretizer._bins[0][0],
                          discretizer._bins[1][1] - discretizer._bins[1][0])
        except Exception:
            epsilon = 0.01

        # figure setup
        xlim = xlim or [0.0, 1.0]
        ylim = ylim or [0.0, 1.0]

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
            color_time_intensity = (idx + 1) / self.memory.n_trajectories
            color = colormap_fn(color_time_intensity)

            states = np.array([traj[ii].state for ii in range(len(traj))])

            if dot_size_means == 'episode_visits':
                sizes = np.array(
                    [traj[ii].n_episode_visits for ii in range(len(traj))]
                )
            elif dot_size_means == 'total_visits':
                raw_states = [traj[ii].raw_state for ii in range(len(traj))]
                sizes = np.array(
                    [
                        np.sum([self.total_visit_counter.count(ss, aa) for aa in range(self.env.action_space.n)])
                        for ss in raw_states
                    ]
                )
            else:
                raise ValueError()

            sizes = 1 + sizes
            sizes = (dot_scale_factor ** 2) * 100 * epsilon * sizes / sizes.max()

            ax.scatter(x=states[:, 0], y=states[:, 1], color=color, s=sizes, alpha=alpha)
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

    def plot_trajectory_actions(self,
                                fignum=None,
                                figsize=(8, 6),
                                n_traj_to_show=10,
                                hide_axis=True,
                                show=True,
                                video_filename=None,
                                colormap_name='Paired',
                                framerate=15,
                                n_skip=1,
                                dot_scale_factor=2.5,
                                alpha=1.0,
                                action_description=None,
                                xlim=None,
                                ylim=None):
        """
        Plot actions (one action = one color) chosen in recent trajectories.

        If video_filename is given, a video file is saved showing the evolution of
        the actions taken in past trajectories.

        Parameters
        ----------
        fignum : str
            Figure name
        figsize : (float, float)
            (width, height) of the image in inches.
        n_traj_to_show : int
            Number of trajectories to visualize in each frame.
        hide_axis : bool
            If True, axes are hidden.
        show : bool
            If True, calls plt.show()
        video_filename : str or None
            If not None, save a video with given filename.
        colormap_name : str, default = 'tab20b'
            Colormap name.
            See https://matplotlib.org/tutorials/colors/colormaps.html
        framerate : int, default: 15
            Video framerate.
        n_skip : int, default: 1
            Skip period: every n_skip trajectories, one trajectory is plotted.
        dot_scale_factor : double
            Scale factor for scatter plot points.
        alpha : float, default: 1.0
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        action_description : list or None (optional)
            List (of strings) containing a description of each action.
            For instance, ['left', 'right', 'up', 'down'].
        xlim: list, default: None
            x plot limits, set to [0, 1] if None
        ylim: list, default: None
            y plot limits, set to [0, 1] if None
        """
        logger.info("Plotting...")

        fignum = fignum or (str(self) + '-actions')
        colormap_fn = plt.get_cmap(colormap_name)
        action_description = action_description or list(range(self.env.action_space.n))

        # discretizer
        try:
            discretizer = self.episode_visit_counter.state_discretizer
            epsilon = min(discretizer._bins[0][1] - discretizer._bins[0][0],
                          discretizer._bins[1][1] - discretizer._bins[1][0])
        except Exception:
            epsilon = 0.01

        # figure setup
        xlim = xlim or [0.0, 1.0]
        ylim = ylim or [0.0, 1.0]

        # indices to visualize
        if video_filename is None:
            indices = [self.memory.n_trajectories - 1]
        else:
            indices = np.arange(self.memory.n_trajectories)[::n_skip]

        # images for video
        images = []

        # for idx in indices:
        for init_idx in indices:
            idx_set = range(max(0, init_idx - n_traj_to_show + 1), init_idx + 1)
            # clear before showing new trajectories
            fig = plt.figure(fignum, figsize=figsize)
            fig.clf()
            canvas = FigureCanvas(fig)
            ax = fig.gca()

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            if hide_axis:
                ax.set_axis_off()

            for idx in idx_set:
                traj = self.memory.trajectories[idx]

                states = np.array([traj[ii].state for ii in range(len(traj))])
                actions = np.array([traj[ii].action for ii in range(len(traj))])

                sizes = (dot_scale_factor ** 2) * 750 * epsilon

                for aa in range(self.env.action_space.n):
                    states_aa = states[actions == aa]
                    color = colormap_fn(aa / self.env.action_space.n)
                    ax.scatter(x=states_aa[:, 0], y=states_aa[:, 1], color=color,
                               s=sizes, alpha=alpha,
                               label=f'action = {action_description[aa]}')

            # for unique legend entries, source: https://stackoverflow.com/a/57600060
            plt.legend(*[*zip(*{l: h for h, l in zip(*ax.get_legend_handles_labels())}.items())][::-1],
                       loc='upper left', bbox_to_anchor=(1.00, 1.00))
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
