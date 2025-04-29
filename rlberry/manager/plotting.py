import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from scipy.stats import norm
import pandas as pd

from rlberry.manager import read_writer_data

import rlberry

logger = rlberry.logger


def plot_writer_data(
    data_source,
    tag,
    xtag=None,
    smooth=False,
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
    return_smoothed_curves=False,
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
        the time elapsed `dw_time_elapsed`, in which case smooth needs to be set to True or there must be only one run.
    smooth : boolean, default=False
        Whether to smooth the curve with a Nadaraya-Watson Kernel smoothing.
        Remark that this also allow for an xtag which is not synchronized on all the simulations (e.g. time for instance).
    smoothing_bandwidth: float or None
        How to choose the bandwidth parameter.
        If float, then smoothing_bandwidth is used directly as a bandwidth.
        If None, a heuristic based on the 10th percentile of nonzero distances in x is used.
    id_agent : int or None, default=None
        id of the agent to plot, if not None plot only the results for the agent whose id is id_agent.
    ax: matplotlib axis or None, default=None
        Matplotlib axis on which we plot. If None, create one. Can be used to
        customize the plot.
    error_representation: str in {"cb", "raw_curves", "ci",  "pi", "none"}
        How to represent multiple simulations. The "ci" and "pi" do not take into account the need for simultaneous inference, it is then harder to draw conclusion from them than with "cb" and "pb" but they are the most widely used.

        - "cb" is a confidence band on the mean curve using functional data analysis (band in which the mean curve is with probability larger than 1-level).

        - "raw curves" is a plot of the raw curves.

        - "pi" is a plot of a non-simultaneous prediction interval with gaussian model around the mean smoothed curve (e.g. we do curve plus/minus gaussian quantile times std).

        - "ci" is a confidence interval with gaussian model around the mean smoothed curve (e.g. we do curve plus/minus gaussian quantile times std divided by sqrt of number of seeds).
        - "none" don't represent the error, only plot the mean smoothed curve.
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
        Warning: this function should return an array of the same size as the input.
    title: str (Optional)
        Optional title to plot. If None, set to tag.
    savefig_fname: str (Optional)
        Name of the figure in which the plot is saved with figure.savefig. If None,
        the figure is not saved.
    return_smoothed_curves: boolean, default=False
        Whether to return a dataframe containing the smoothed curves. If True,
        returns the tuple (data_preprocessed, data_smoothed).
    linestyles: boolean, default=False
        Whether to use different linestyles for each curve.
    Returns
    -------
    Pandas DataFrame with processed data.

    Examples
    --------
    >>> from rlberry_research.agents.torch import A2CAgent, DQNAgent
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
        data_source,
        many_agent_by_str_datasource=False,
        preprocess_tag=tag,
        preprocess_func=preprocess_func,
        id_agent=id_agent,
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
    data.loc[:, "n_simu"] = data["n_simu"].astype(int)
    if sub_sample:
        new_df = pd.DataFrame()
        for name in data["name"].unique():
            n_simu_tot = int(data.loc[data["name"] == name, "n_simu"].max()) + 1
            for simu in range(n_simu_tot):
                df_name_simu = data.loc[
                    (data["n_simu"] == simu) & (data["name"] == name)
                ]
                step = len(df_name_simu) // 1000
                if len(df_name_simu) > 0:
                    if step > 1:
                        df_sub = df_name_simu.sort_values(by=xtag).iloc[
                            ::step
                        ]  # do the sub-sampling
                        new_df = pd.concat([new_df, df_sub], ignore_index=True)
                    else:
                        new_df = pd.concat([new_df, df_name_simu], ignore_index=True)
        data = new_df

    if ax is None:
        figure, ax = plt.subplots(1, 1)
    if smooth:
        data_smoothed = plot_curves_smoothed_NW(
            data[["name", xtag, "value", "n_simu"]],
            xtag,
            "value",
            smoothing_bandwidth,
            ax,
            error_representation,
            n_boot,
            level,
            False,
            None,
            linestyles,
        )
    else:
        data_smoothed = plot_curves_with_same_x(
            data[["name", xtag, "value", "n_simu"]],
            xtag,
            "value",
            ax,
            error_representation,
            level,
            False,
            None,
            linestyles,
        )
    ax.set_xlabel(xtag)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if savefig_fname is not None:
        plt.gcf().savefig(savefig_fname)
    if show:
        plt.show()
    if return_smoothed_curves:
        return data, data_smoothed
    else:
        return data


def plot_curves_smoothed_NW(
    data,
    x,
    y,
    smoothing_bandwidth=None,
    ax=None,
    error_representation="ci",
    n_boot=2500,
    level=0.9,
    show=True,
    savefig_fname=None,
    linestyles=False,
):
    """
    Plot the performances contained in the data (see data parameter to learn what format it should be).

    If there are several simulations, an error band is plotted.

    In all cases a smoothing is performed.

    Parameters
    ----------
    data: a pandas dataframe
        data must contain the columns "name", "n_simu", an x column and a y column.

        - "n_simu" contain the simulation number (e.g. the seed) for which the raw is computed (beginning at 0 until the total number of seeds).

        - "name" is the name of the algorithm for which the raw is computed.

        - x column is named according to x parameter and contain values to have in x axis.

        - y column is named according to y parameter and contain values to have in y axis.

    smoothing_bandwidth: float or None
        How to choose the bandwidth parameter.
        If float, then smoothing_bandwidth is used directly as a bandwidth.
        If None, a heuristic based on the 10th percentile of nonzero distances in x is used.
    ax: matplotlib axis or None, default=None
        Matplotlib axis on which we plot. If None, create one. Can be used to
        customize the plot.
    error_representation: str in {"cb", "raw_curves", "ci",  "pi", "none"}
        How to represent multiple simulations. The "ci" and "pi" do not take into account the need for simultaneous inference, it is then harder to draw conclusion from them than with "cb" but they are the most widely used.

        - "cb" is a confidence band on the mean curve using functional data analysis (band in which the mean curve is with probability larger than 1-level). Method from [1].

        - "raw curves" is a plot of the raw curves.

        - "pi" is a plot of a non-simultaneous prediction interval with gaussian model around the mean smoothed curve (e.g. we do curve plus/minus gaussian quantile times std).

        - "ci" is a confidence interval with gaussian model around the mean smoothed curve (e.g. we do curve plus/minus gaussian quantile times std divided by sqrt of number of seeds).
        - "none" don't represent the error, only plot the mean smoothed curve.

    n_boot: int, default=2500,
        Number of bootstrap evaluations used for confidence interval estimation.
        Only used if error_representation = "ci".
    level: float, default=0.95,
        Level of the confidence (or prediction) interval. Only used if error_representation is not "raw_curves".
    show: bool, default=True
        If true, calls plt.show().
    savefig_fname: str (Optional)
        Name of the figure in which the plot is saved with figure.savefig. If None,
        the figure is not saved.
    linestyles: boolean, default=False
        Whether to use different linestyles for each curve.

    Examples
    --------
    >>> import pandas as pd
    >>> from rlberry.manager import plot_smoothed_curve
    >>>  df = pd.DataFrame(
        {"name": ["a", "a", "a"], "x": [1, 2, 3], "y": [3, 4, 5], "n_simu": [0, 0, 0]}
    )
    >>> plot_smoothed_curve(df, "x", "y")

    References
    ----------
        [1] Degras, D. (2017). Simultaneous confidence bands for the mean of functional data. Wiley Interdisciplinary Reviews: Computational Statistics, 9(3), e1397.

    """

    xlabel = x
    ylabel = y
    x_values = data[xlabel].values
    min_x, max_x = x_values.min(), x_values.max()
    xplot = np.linspace(min_x, max_x, 500, endpoint=True)

    ax, styles, cmap = _prepare_ax(data, ax, linestyles)

    def process(df):
        """
        Nadaraya-Watson kernel smoothing
        """
        n_tot_simu = int(df["n_simu"].max()) + 1

        Yhat = np.zeros([n_tot_simu, len(xplot)])
        bw = smoothing_bandwidth
        for f in range(n_tot_simu):
            Y = df_name.loc[df["n_simu"] == f, ylabel].values

            try:
                np.isfinite(Y)
            except:
                raise ValueError("non-finite (or non float) data detected.")

            if not np.all(np.isfinite(Y)):
                logger.warning(
                    "Some of the values are not finite. Not plotting the associated curves."
                )
                Yhat[f] = np.nan
            else:
                X = df_name.loc[df["n_simu"] == f, xlabel].values.astype(float)
                if len(X) != 0:
                    nw = Smoothed_curve_NW(X, xplot, bandwidth=bw)
                    Yhat[f] = nw.get_y_smoothed(Y)
                else:
                    Yhat[f] = np.nan * np.ones(len(xplot))

        return Yhat

    names = np.unique(data["name"])
    data_smoothed = pd.DataFrame()

    for id_c, name in enumerate(names):
        df_name = data.loc[data["name"] == name]
        n_tot_simu = int(df_name["n_simu"].max()) + 1
        Xhat = process(df_name)
        mu = np.nanmean(Xhat, axis=0)
        id_plot = xplot <= np.max(df_name[xlabel])

        ax.plot(
            xplot[id_plot],
            mu[id_plot],
            label=name,
            color=cmap[id_c],
            linestyle=(0, styles[id_c]),
        )
        data_smoothed = pd.concat(
            [
                data_smoothed,
                pd.DataFrame(
                    {
                        "name": [name] * len(id_plot),
                        "x": xplot[id_plot],
                        "y": mu[id_plot],
                    }
                ),
            ],
            ignore_index=True,
        )

        if (error_representation == "raw_curves") and (n_tot_simu > 1):
            for n_simu in range(n_tot_simu):
                x_simu = df_name.loc[df_name["n_simu"] == n_simu, xlabel].values.astype(
                    float
                )
                y = df_name.loc[df_name["n_simu"] == n_simu, ylabel].values
                if n_simu == 0:
                    ax.plot(x_simu, y, alpha=0.2, label="raw " + name, color=cmap[id_c])
                else:
                    ax.plot(x_simu, y, alpha=0.25, color=cmap[id_c])
        elif n_tot_simu > 1:
            sigma = np.sqrt(np.sum((Xhat - mu) ** 2, axis=0) / (len(Xhat) - 1))

            if error_representation == "ci":
                quantile = norm.ppf(1 - (1 - level) / 2)
                y_err = quantile * sigma / np.sqrt(n_tot_simu)

            elif error_representation == "pi":
                quantile = norm.ppf(1 - (1 - level) / 2)
                y_err = quantile * sigma

            elif error_representation == "cb":
                if n_tot_simu < 1 / (1 - level):
                    logger.warning(
                        "Computing a cb that cannot achieve the level prescribed because there are not enough seeds."
                    )

                res = []
                # Bootstrap estimation of confidence interval
                if not (np.all(sigma == 0)):
                    for b in range(n_boot):
                        id_b = (
                            np.random.choice(n_tot_simu, size=n_tot_simu, replace=True)
                            - 1
                        )
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
                    y_err = np.zeros(len(xplot))
                    logger.warning(
                        "The variance of the curve was 0, the confidence bound is very biased"
                    )
            elif error_representation == "none":
                pass
            else:
                raise ValueError("error_representation not implemented")
            if error_representation != "none":
                ax.fill_between(
                    xplot[id_plot],
                    mu.ravel()[id_plot] - y_err[id_plot],
                    mu.ravel()[id_plot] + y_err[id_plot],
                    alpha=0.25,
                    color=cmap[id_c],
                )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if show:
        plt.show()
    if savefig_fname is not None:
        plt.gcf().savefig(savefig_fname)

    return data_smoothed


def plot_curves_with_same_x(
    data,
    x,
    y,
    ax=None,
    error_representation="pi",
    level=0.9,
    show=True,
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

        - x column is named according to x parameter and contain values to have in x axis.

        - y column is named according to y parameter and contain values to have in y axis.

    ax: matplotlib axis or None, default=None
        Matplotlib axis on which we plot. If None, create one. Can be used to
        customize the plot.
    error_representation: str in {"raw_curves", "ci",  "pi", "none"}, default="pi"
        How to represent multiple simulations.

        - "raw curves" is a plot of the raw curves.

        - "pi" is a plot of a non-simultaneous prediction interval with gaussian model around the mean curve (e.g. we do curve plus/minus gaussian quantile times std).

        - "ci" is a confidence interval on the prediction interval with gaussian model around the mean curve (e.g. we do curve plus/minus gaussian quantile times std divided by sqrt of number of seeds).
        - "none" don't represent the error, only plot the mean smoothed curve.

    level: float, default=0.95,
        Level of the confidence (or prediction) interval. Only used if error_representation is not "raw_curves".
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

    """
    xlabel = x
    ylabel = y
    assert len(data) > 0, "dataset is empty"
    n_tot_simu = int(data["n_simu"].max())

    # check that every simulation have the same xs or truncate
    processed_df = pd.DataFrame()
    for name in np.unique(data["name"]):
        df_name = data.loc[data["name"] == name]
        x_simu_0 = df_name.loc[df_name["n_simu"] == 0, xlabel].values.astype(float)
        for n_simu in range(1, int(n_tot_simu)):
            x_simu = df_name.loc[df_name["n_simu"] == n_simu, xlabel].values.astype(
                float
            )
            if len(x_simu) != len(x_simu_0):
                logger.warning("x axis is not the same for all the runs, truncating.")
            x_simu_0 = np.intersect1d(x_simu_0, x_simu)
        df_name = df_name.loc[df_name[xlabel].apply(lambda x: x in x_simu_0)]
        assert (
            len(df_name) > 0
        ), "x_axis are incompatible across runs, you should use smoothing"
        processed_df = pd.concat([processed_df, df_name], ignore_index=True)
    data = processed_df

    ax, styles, cmap = _prepare_ax(data, ax, linestyles)

    names = np.unique(data["name"])
    data_smoothed = pd.DataFrame()
    for id_c, name in enumerate(names):
        df_name = data.loc[data["name"] == name, [xlabel, ylabel, "n_simu"]]
        x_plot = df_name.loc[df_name["n_simu"] == 0, xlabel].values.astype(float)

        y_mean = (
            df_name[[xlabel, ylabel]]
            .groupby([xlabel])
            .mean()
            .values.astype(float)
            .ravel()
        )
        y_std = (
            df_name[[xlabel, ylabel]]
            .groupby([xlabel])
            .std()
            .values.astype(float)
            .ravel()
        )

        quantile = norm.ppf(1 - (1 - level) / 2)
        ax.plot(x_plot, y_mean, color=cmap[id_c], label=name)
        data_smoothed = pd.concat(
            [
                data_smoothed,
                pd.DataFrame({"name": [name] * len(x_plot), "x": x_plot, "y": y_mean}),
            ],
            ignore_index=True,
        )

        if error_representation in ["ci", "pi"]:
            if error_representation == "pi":
                y_err = quantile * y_std
            else:
                y_err = quantile * y_std / np.sqrt(n_tot_simu)

            ax.fill_between(
                x_plot,
                y_mean - y_err,
                y_mean + y_err,
                alpha=0.25,
                color=cmap[id_c],
            )
        elif error_representation == "raw_curves":
            for n_simu in range(n_tot_simu + 1):
                x_simu = df_name.loc[df_name["n_simu"] == n_simu, xlabel].values.astype(
                    float
                )
                y = df_name.loc[df_name["n_simu"] == n_simu, ylabel].values

                if n_simu == 0:
                    ax.plot(x_simu, y, alpha=0.2, color=cmap[id_c])
                else:
                    ax.plot(x_simu, y, alpha=0.25, color=cmap[id_c])
        elif error_representation == "none":
            pass
        else:
            raise ValueError(
                "Error representation {} not known for non-smoothed plots".format(
                    error_representation
                )
            )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if show:
        plt.show()

    if savefig_fname is not None:
        plt.gcf().savefig(savefig_fname)

    return data_smoothed


def _prepare_ax(data, ax, linestyles):
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

    names = data["name"].unique()
    if len(names) <= 10:
        cmap = plt.cm.tab10.colors[: len(names)]
    else:
        cmap = [plt.cm.gist_rainbow(i / len(names)) for i in range(len(names))]

    return ax, styles, cmap


class Smoothed_curve_NW:
    """
    Nadaraya-Watson kernel smoothing

    Parameters
    ----------
    X: array of floats
        Observed x-axis coordinates, usually either global_step or time.
    xref: array of floats
        x values at which we want to compute the smoothed curve
    bandwidth: float or None, default=None
        Bandwidth parameter which corresponds to the width of a window on which to smooth for Gaussian kernel,
        if None, use the 10th percentile of the nonzero distances between all X[i]

    """

    def __init__(self, X, xref, bandwidth=None):
        self.kernel = lambda x: np.exp(-(x**2) / 2)
        self.bandwidth = bandwidth
        self.Hmatrix = self.H(X, xref)

    def H(self, xi, xref):
        D = np.abs((xi[:, None] - xref).T)
        nonzero_distances = D.ravel()[D.ravel() > 0]
        if len(nonzero_distances) == 0:
            bandwidth = (np.max(xi) - np.min(xi)) / 100
        else:
            bandwidth = (
                float(np.percentile(nonzero_distances, 10))
                if self.bandwidth is None
                else self.bandwidth
            )
        numerator = self.kernel(D / bandwidth)
        return numerator / np.sum(numerator, axis=1)[:, np.newaxis]

    def get_y_smoothed(self, y):
        return self.Hmatrix.dot(y)
