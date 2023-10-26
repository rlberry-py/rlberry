import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import numbers
from scipy.stats import norm
import pandas as pd

from rlberry.manager import read_writer_data

from skfda.representation.grid import FDataGrid
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.preprocessing.smoothing.validation import SmoothingParameterSearch

import rlberry

logger = rlberry.logger


def plot_writer_data(
    data_source,
    tag,
    xtag=None,
    smooth=True,
    smoothing_bandwidth=None,
    id_agent=None,
    ax=None,
    error_representation="ci",
    n_boot=500,
    level=0.9,
    sub_sample=True,
    show=True,
    preprocess_func=None,
    title=None,
    savefig_fname=None,
    linestyles=False,
):
    """
    Given a list of ExperimentManager or a folder, plot data (corresponding to info) obtained in each episode.
    The dictionary returned by agents' .fit() method must contain a key equal to `info`.

    If there are several simulations, a confidence interval is plotted. In all cases a smoothing is performed

    Parameters
    ----------
    data_source : :class:`~rlberry.manager.ExperimentManager`, or list of :class:`~rlberry.manager.ExperimentManager` or str or list of str
        - If ExperimentManager or list of ExperimentManager, load data from it (the agents must be fitted).

        - If str, the string must be the string path of a directory,  each
        subdirectory of this directory must contain pickle files.
        load the data from the directory of the latest experiment in date.
        This str should be equal to the value of the `output_dir` parameter in
        :class:`~rlberry.manager.ExperimentManager`.

        - If list of str, each string must be a directory containing pickle files
        load the data from these pickle files.

        Note: the agent's save function must save its writer at the key `_writer`.
        This is the default for rlberry agents.
    tag : str
        Tag of data to plot on y-axis.
    xtag : str or None, default=None
        Tag of data to plot on x-axis. If None, use 'global_step'. Another often-used x-axis is
        the time elapsed `dw_time_elapsed`.
    smooth : boolean, default=True
        Whether to smooth the curve with a Nadaraya-Watson Kernel smoothing.
    smoothing_bandwidth: float or array of floats or None
        How to choose the bandwidth parameter. If float, then smoothing_bandwidth is used
        directly as a bandwidth and if is an array, a parameter search using smoothing_bandwidth is
        used if None, a parameter search from a range of 20 possible values choosen by heuristics is performed.
    id_agent : int or None, default=None
        id of the agent to plot, if not None plot only the results for the agent whose id is id_agent.
    ax: matplotlib axis or None, default=None
        Matplotlib axis on which we plot. If None, create one. Can be used to
        customize the plot.
    error_representation: str in {"cb", "raw_curves", "ci",  "pi"}
        How to represent multiple simulations. The "ci" and "pi" do not take into account the need for simultaneous inference, it is then harder to draw conclusion from them than with "cb" and "pb" but they are the most widely used.
        - "cb" is a confidence band on the mean curve using functional data analysis (band in which the curve is with probability larger than 1-level).
        - "raw curves" is a plot of the raw curves.
        - "pi" is a plot of a non-simultaneous prediction interval with gaussian model around the mean smoothed curve (e.g. we do curve plus/minus gaussian quantile times std).
        - "ci" is a confidence interval on the prediction interval with gaussian model around the mean smoothed curve (e.g. we do curve plus/minus gaussian quantile times std divided by sqrt of number of seeds).
    n_boot: int, default=500,
        Number of bootstrap evaluations used for confidence interval estimation.
        Only used if error_representation = "ci".
    level: float, default=0.95,
        Level of the confidence interval. Only used if error_representation = "ci"
    sub_sample, boolean, default = True,
        If True, use up to 1000 points for one given seed of one agent to reduce computational cost.
    show: bool, default=True
        If True, calls plt.show().
    preprocess_func: Callable, default=None
        Function to apply to 'tag' column before plot. For instance, if tag=episode_rewards,
        setting preprocess_func=np.cumsum will plot cumulative rewards. If None, do nothing.
    title: str (Optional)
        Optional title to plot. If None, set to tag.
    savefig_fname: str (Optional)
        Name of the figure in which the plot is saved with figure.savefig. If None,
        the figure is not saved.
    linestyles: boolean, default=False
        Whether to use different linestyles for each curve.
    Returns
    -------
    Pandas DataFrame with processed data.

    Examples
    --------
    >>> from rlberry.agents.torch import A2CAgent, DQNAgent
    >>> from rlberry.manager import ExperimentManager, plot_writer_data
    >>> from rlberry.envs import gym_make
    >>>
    >>> if __name__=="__main__":
    >>>     managers = [ ExperimentManager(
    >>>         agent_class,
    >>>         (gym_make, dict(id="CartPole-v1")),
    >>>         fit_budget=4e4,
    >>>         eval_kwargs=dict(eval_horizon=500),
    >>>         n_fit=1,
    >>>         parallelization="process",
    >>>         mp_context="spawn",
    >>>         seed=42,
    >>>          ) for agent_class in [A2CAgent, DQNAgent]]
    >>>     for manager in managers:
    >>>         manager.fit()
    >>>     # We have only one seed (n_fit=1) hence the curves are automatically smoothed
    >>>     data = plot_writer_data(managers, "episode_rewards")
    """
    title = title or tag
    if preprocess_func is not None:
        ylabel = "value"
    else:
        ylabel = tag
    processed_df = read_writer_data(
        data_source, tag, preprocess_func, id_agent=id_agent
    )

    data = processed_df[processed_df["tag"] == tag]

    if len(data) == 0:
        logger.error("[plot_writer_data]: No data to be plotted.")
        return

    if xtag is None:
        xtag = "global_step"

    if data[xtag].notnull().sum() > 0:
        if data[xtag].isna().sum() > 0:
            logger.warning(
                f"Plotting {tag} vs {xtag}, but {xtag} might be missing for some agents."
            )
    else:
        data[xtag] = data.index

    if sub_sample:
        new_df = pd.DataFrame()
        for name in data["name"].unique():
            n_simu_tot = int(data.loc[data["name"] == name, "n_simu"].max())
            for simu in range(n_simu_tot):
                df_name_simu = data.loc[
                    (data["n_simu"] == simu) & (data["name"] == name)
                ]
                step = len(df_name_simu) // 1000
                if step > 1:
                    df_sub = df_name_simu.sort_values(by=xx).iloc[
                        ::step
                    ]  # do the sub-sampling
                    new_df = pd.concat([new_df, df_sub], ignore_index=True)
                else:
                    new_df = pd.concat([new_df, df_name_simu], ignore_index=True)
    data = new_df
    if ax is None:
        figure, ax = plt.subplots(1, 1)
    plot_smoothed_curve(
        data[["name", xtag, "value", "n_simu"]],
        xtag,
        "value",
        smooth,
        smoothing_bandwidth,
        ax,
        error_representation,
        n_boot,
        level,
        False,
        title,
        None,
        linestyles,
    )

    ax.set_xlabel(xx)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if savefig_fname is not None:
        fig.savefig(savefig_fname)
    if show:
        plt.show()
    return data


def plot_smoothed_curve(
    data,
    x,
    y,
    smooth=True,
    smoothing_bandwidth=None,
    ax=None,
    error_representation="ci",
    n_boot=2500,
    level=0.9,
    show=True,
    title=None,
    savefig_fname=None,
    linestyles=False,
):
    """
    Plot the performances contained in the data (see data parameter to learn what format it should be).

    If there are several simulations, a confidence interval is plotted.

    In all cases a smoothing is performed

    Parameters
    ----------
    data: a pandas dataframe
        data must contain the columns "name", "n_simu", an x column and a y column.
        - "n_simu" contain the simulation number (e.g. the seed) for which the raw is computed.
        - "name" is the name of the algorithm for which the raw is computed.
        - x column is named according to x parameter and contain values to have in x axis
        - y column is named according to y parameter and contain values to have in y axis

    smooth : boolean, default=True
        Whether to smooth the curve with a Nadaraya-Watson Kernel smoothing.
    smoothing_bandwidth: float or array of floats or None
        How to choose the bandwidth parameter. If float, then smoothing_bandwidth is used
        directly as a bandwidth and if is an array, a parameter search using smoothing_bandwidth is
        used if None, a parameter search from a range of 20 possible values choosen by heuristics is performed.
    ax: matplotlib axis or None, default=None
        Matplotlib axis on which we plot. If None, create one. Can be used to
        customize the plot.
    error_representation: str in {"cb", "raw_curves", "ci",  "pi"}
        How to represent multiple simulations. The "ci" and "pi" do not take into account the need for simultaneous inference, it is then harder to draw conclusion from them than with "cb" and "pb" but they are the most widely used.
        - "cb" is a confidence band on the mean curve using functional data analysis (band in which the curve is with probability larger than 1-level). Method from [1], using scikit-fda [2] library.
        - "raw curves" is a plot of the raw curves.
        - "pi" is a plot of a non-simultaneous prediction interval with gaussian model around the mean smoothed curve (e.g. we do curve plus/minus gaussian quantile times std).
        - "ci" is a confidence interval on the prediction interval with gaussian model around the mean smoothed curve (e.g. we do curve plus/minus gaussian quantile times std divided by sqrt of number of seeds).
    n_boot: int, default=2500,
        Number of bootstrap evaluations used for confidence interval estimation.
        Only used if error_representation = "ci".
    level: float, default=0.95,
        Level of the confidence interval. Only used if error_representation = "ci"
    show: bool, default=True
        If true, calls plt.show().
    savefig_fname: str (Optional)
        Name of the figure in which the plot is saved with figure.savefig. If None,
        the figure is not saved.
    linestyles: boolean, default=False
        Whether to use different linestyles for each curve.

    References
    ----------
        [1] Degras, D. (2017). Simultaneous confidence bands for the mean of functional data. Wiley Interdisciplinary Reviews: Computational Statistics, 9(3), e1397.
        [2] scikit-fda, Carlos Ramos Carreño, hzzhyj, mellamansanchez, Pablo Marcos, pedrorponga, David del Val, Pablo, David García Fernández, Martín, Miguel Carbajo Berrocal, ElenaPetrunina, Pablo Cuesta Sierra, Rafa Hidalgo, Clément Lejeune, amandaher, dSerna4, ego-thales, pedrog99, Jorge Duque, … Álvaro Castillo. (2023). GAA-UAM/scikit-fda: Version 0.9 (0.9). Zenodo. https://doi.org/10.5281/zenodo.10016930

    """
    xlabel = x
    ylabel = y
    x_values = data[xlabel].values
    min_x, max_x = x_values.min(), x_values.max()

    if not isinstance(smoothing_bandwidth, numbers.Number):
        sorted_x = np.sort(np.unique(x_values))
        if len(sorted_x) > 200:
            # second percentile minus minimum
            min_bandwidth_x = (sorted_x[1] - sorted_x[0]) * 3
        else:
            min_bandwidth_x = sorted_x[1] - sorted_x[0]
    xplot = np.linspace(min_x, max_x, 500, endpoint=True)

    if ax is None:
        figure, ax = plt.subplots(1, 1)

    # Customizing linestyle
    if linestyles:
        # Number of unique dash styles. Default: 4 styles max.
        linestyles = ["", (1, 1), (5, 5), (1, 5, 3, 5)]
        # Cycle through default linestyles.
        dash_cycler = cycle(linestyles)
        styles = [next(dash_cycler) for _ in range(data["name"].unique().size)]
    else:
        styles = [() for _ in range(data["name"].unique().size)]

    n_tot_simu = int(data["n_simu"].max())
    names = data["name"].unique()
    if len(names) <= 10:
        cmap = plt.cm.tab10.colors[: len(names)]
    else:
        cmap = [plt.cm.gist_rainbow(i / len(names)) for i in range(len(names))]

    def process(df):
        """
        Change shape and smooth the curves contained in the dataset df if necessary.
        """
        # Nadaraya-Watson kernel smoothing
        # with cross validation bandwidth selection
        if not isinstance(smoothing_bandwidth, numbers.Number):
            if smoothing_bandwidth is None:
                bandwidth = np.linspace(min_bandwidth_x, (max_x - min_x) / 100, 10)
            else:
                bandwidth = smoothing_bandwidth
            nw = SmoothingParameterSearch(
                KernelSmoother(
                    kernel_estimator=NadarayaWatsonHatMatrix(), output_points=xplot
                ),
                bandwidth,
                param_name="kernel_estimator__bandwidth",
            )
        else:
            nw = KernelSmoother(
                kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=smoothing_bandwidth),
                output_points=xplot,
            )

        Xhat = np.zeros([n_tot_simu, len(xplot)])
        bw = False
        for f in range(n_tot_simu):
            X = df_name.loc[df["n_simu"] == f, ylabel].values
            if smooth and (bw is False):
                X_grid = df_name.loc[df["n_simu"] == f, xlabel].values.astype(float)
                fd = FDataGrid([X], X_grid, domain_range=((min_x, max_x),))
                nw.fit(fd)  # Find the smoothing bandwidth once
                bw = True  # don't search for bandwidth in futur run, reuse
                Xhat[f] = nw.transform(fd).data_matrix.ravel()  # apply smoothing
            elif smooth:
                X_grid = df_name.loc[df["n_simu"] == f, xlabel].values.astype(float)
                fd = FDataGrid([X], X_grid, domain_range=((min_x, max_x),))
                Xhat[f] = nw.transform(fd).data_matrix.ravel()  # apply smoothing
            else:
                Xhat[f] = X

        return Xhat

    for id_c, name in enumerate(names):
        df_name = data.loc[data["name"] == name]

        processed = process(df_name).mean(axis=0)

        id_plot = xplot <= np.max(df_name[xlabel])

        ax.plot(
            xplot[id_plot],
            processed[id_plot],
            label=name,
            color=cmap[id_c],
            linestyle=(0, styles[id_c]),
        )

        if error_representation == "raw_curves":
            for n_simu in range(n_tot_simu):
                x_simu = df_name.loc[df_name["n_simu"] == n_simu, xlabel].values.astype(
                    float
                )
                y = df_name.loc[df_name["n_simu"] == n_simu, ylabel].values
                if n_simu == 0:
                    ax.plot(x_simu, y, alpha=0.2, label="raw " + name, color=cmap[id_c])
                else:
                    ax.plot(x_simu, y, alpha=0.25, color=cmap[id_c])
        else:
            if smooth == False:
                raise ValueError(
                    "In order to use errorbar = {} you need smooth = True".format(
                        error_representation
                    )
                )

            Xhat = process(df_name)
            mu = np.mean(Xhat, axis=0)
            sigma = np.sqrt(np.sum((Xhat - mu) ** 2, axis=0) / (len(Xhat) - 1))

            if error_representation == "ci":
                quantile = norm.ppf(1 - (1 - level) / 2)
                y_err = quantile * sigma / np.sqrt(n_tot_simu)

            elif error_representation == "pi":
                quantile = norm.ppf(1 - (1 - level) / 2)
                y_err = quantile * sigma

            elif error_representation == "cb":
                if n_tot_simu < 1 / (1 - level):
                    logger.warn(
                        "Computing a cb that cannot achieve the level prescribed because there are not enough seeds."
                    )

                res = []
                # Bootstrap estimation of confidence interval

                for b in range(n_boot):
                    id_b = np.random.choice(n_tot_simu, size=n_tot_simu, replace=True)
                    mustar = np.mean(Xhat[id_b], axis=0)
                    residus = (
                        np.sqrt(len(xplot[sigma != 0]))
                        / sigma[sigma != 0]
                        * np.abs(mustar[sigma != 0] - mu[sigma != 0])
                    )
                    res.append(np.max(residus))

                y_err = (
                    sigma.ravel()
                    / np.sqrt(len(xplot))
                    * np.quantile(res, 1 - (1 - level) / 2)
                )
            else:
                raise ValueError("error_representation not implemented")

            ax.fill_between(
                xplot[id_plot],
                mu.ravel()[id_plot] - y_err[id_plot],
                mu.ravel()[id_plot] + y_err[id_plot],
                alpha=0.25,
                color=cmap[id_c],
            )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.legend()

    if show:
        plt.show()

    if savefig_fname is not None:
        plt.gcf().savefig(savefig_fname)

    return data
