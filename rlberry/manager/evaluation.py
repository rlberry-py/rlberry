import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pickle
import bz2
import _pickle as cPickle
from itertools import cycle
import dill
from distutils.version import LooseVersion

from rlberry.manager import AgentManager
import rlberry


logger = rlberry.logger


def evaluate_agents(
    agent_manager_list,
    n_simulations=5,
    choose_random_agents=True,
    fignum=None,
    show=True,
    plot=True,
    sns_kwargs=None,
):
    """
    Evaluate and compare each of the agents in agent_manager_list.

    Parameters
    ----------
    agent_manager_list : list of AgentManager objects.
    n_simulations: int
        Number of calls to the eval() method of each AgentManager instance.
    choose_random_agents: bool
        If true and n_fit>1, use a random fitted agent from each AgentManager at each evaluation.
        Otherwise, each fitted agent of each AgentManager is evaluated n_simulations times.
    fignum: string or int
        Identifier of plot figure.
    show: bool
        If true, calls plt.show().
    plot: bool
        If false, do not plot.
    sns_kwargs:
        Extra parameters for sns.boxplot

    Returns
    -------
    dataframe with the evaluation results.

    Examples
    --------
    >>> from rlberry.agents.torch import A2CAgent, DQNAgent
    >>> from rlberry.manager import AgentManager, evaluate_agents
    >>> from rlberry.envs import gym_make
    >>>
    >>> if __name__=="__main__":
    >>>     managers = [ AgentManager(
    >>>         agent_class,
    >>>         (gym_make, dict(id="CartPole-v1")),
    >>>         fit_budget=1e4,
    >>>         eval_kwargs=dict(eval_horizon=500),
    >>>         n_fit=1,
    >>>         parallelization="process",
    >>>         mp_context="spawn",
    >>>         seed=42,
    >>>          ) for agent_class in [A2CAgent, DQNAgent]]
    >>>     for manager in managers:
    >>>         manager.fit()
    >>>     data = evaluate_agents(managers, n_simulations=50, plot=False)

    """
    sns_kwargs = sns_kwargs or {}

    #
    # evaluation
    #

    eval_outputs = []
    for agent_manager in agent_manager_list:
        logger.info(f"Evaluating {agent_manager.agent_name}...")
        if choose_random_agents:
            outputs = agent_manager.eval_agents(n_simulations)
        else:
            outputs = []
            for idx in range(len(agent_manager.agent_handlers)):
                outputs += list(agent_manager.eval_agents(n_simulations, agent_id=idx))

        if len(outputs) > 0:
            eval_outputs.append(outputs)

    if len(eval_outputs) == 0:
        logger.error(
            "[evaluate_agents]: No evaluation data. Make sure AgentManager.fit() has been called."
        )
        return

    #
    # plot
    #

    # build unique agent IDs (in case there are two agents with the same ID)
    unique_ids = []
    id_count = {}
    for agent_manager in agent_manager_list:
        name = agent_manager.agent_name
        if name not in id_count:
            id_count[name] = 1
        else:
            id_count[name] += 1

        unique_ids.append(name + "*" * (id_count[name] - 1))

    # convert output to DataFrame
    data = {}
    for agent_id, out in zip(unique_ids, eval_outputs):
        data[agent_id] = out
    output = pd.DataFrame(data)

    # plot
    if plot:
        plt.figure(fignum)
        with sns.axes_style("whitegrid"):
            ax = sns.boxplot(data=output, **sns_kwargs)
            ax.set_xlabel("agent")
            ax.set_ylabel("evaluation output")
            if show:
                plt.show()

    return output


def read_writer_data(data_source, tag=None, preprocess_func=None, id_agent=None):
    """
    Given a list of AgentManager or a folder, read data (corresponding to info) obtained in each episode.
    The dictionary returned by agents' .fit() method must contain a key equal to `info`.

    Parameters
    ----------
    data_source : :class:`~rlberry.manager.AgentManager`, or list of :class:`~rlberry.manager.AgentManager` or str or list of str
        - If AgentManager or list of AgentManager, load data from it (the agents must be fitted).

        - If str, the string must be the string path of a directory,  each
        subdirectory of this directory must contain pickle files.
        Load the data from the directory of the latest experiment in date.
        This str should be equal to the value of the `output_dir` parameter in
        :class:`~rlberry.manager.AgentManager`.

        - If list of str, each string must be a directory containing pickle files.
        Load the data from these pickle files.

        Note: the agent's save function must save its writer at the key `_writer`.
        This is the default for rlberry agents.

    tag :  str or list of str or None
        Tag of data that we want to preprocess.

    preprocess_func: Callable or None or list of Callable or None
        Function to apply to 'tag' column before returning data.
        For instance, if tag=episode_rewards,setting preprocess_func=np.cumsum
        will return cumulative rewards
        If None, do not preprocess.
        If tag is a list, preprocess_func must be None or a list of Callable or
        None that matches the length of tag.

    id_agent: int or None, default=None
        If not None, returns the data only for agent 'id_agent'.

    Returns
    -------
    Pandas DataFrame with data from writers.

    Examples
    --------
    >>> from rlberry.agents.torch import A2CAgent, DQNAgent
    >>> from rlberry.manager import AgentManager, read_writer_data
    >>> from rlberry.envs import gym_make
    >>>
    >>> if __name__=="__main__":
    >>>     managers = [ AgentManager(
    >>>         agent_class,
    >>>         (gym_make, dict(id="CartPole-v1")),
    >>>         fit_budget=1e4,
    >>>         eval_kwargs=dict(eval_horizon=500),
    >>>         n_fit=1,
    >>>         parallelization="process",
    >>>         mp_context="spawn",
    >>>         seed=42,
    >>>          ) for agent_class in [A2CAgent, DQNAgent]]
    >>>     for manager in managers:
    >>>         manager.fit()
    >>>     data = read_writer_data(managers)
    """
    input_dir = None

    if not isinstance(data_source, list):
        if isinstance(data_source, AgentManager):
            data_source = [data_source]
        else:
            take_last_date = True
    else:
        if not isinstance(data_source[0], AgentManager):
            take_last_date = False
            for dir in data_source:
                files = list(Path(dir).iterdir())
                if len(files) == 0:
                    raise RuntimeError(
                        "One of the files in data_source does not contain pickle files"
                    )

    if isinstance(data_source[0], AgentManager):
        agent_manager_list = data_source
    else:
        input_dir = data_source

    if isinstance(tag, str):
        tags = [tag]
        preprocess_funcs = [preprocess_func or (lambda x: x)]
    elif tag is not None:
        tags = tag
        if preprocess_func is None:
            preprocess_funcs = [lambda x: x for _ in range(len(tags))]
        else:
            assert len(preprocess_func) == len(tags)
            preprocess_funcs = preprocess_func

    writer_datas = []
    if input_dir is not None:
        if take_last_date:
            subdirs = list((Path(input_dir) / "manager_data").iterdir())
            agent_name_list = [str(p.stem).split("_")[0] for p in subdirs]
            for name in agent_name_list:
                filename, dir_name = _get_last_xp(input_dir, name)
                writer_datas.append(_load_data(filename, dir_name, id_agent))
        else:
            agent_name_list = [str(Path(p).stem).split("_")[0] for p in input_dir]
            agent_dirs = [str(Path(p).parent).split("_")[0] for p in input_dir]

            for id_f, filename in enumerate(input_dir):
                writer_datas.append(_load_data(filename, agent_dirs[id_f], id_agent))
    else:
        for manager in agent_manager_list:
            # Important: since manager can be a RemoteAgentManager,
            # it is important to avoid repeated accesses to its methods and properties.
            # That is why writer_data is taken from the manager instance only in
            # the line below.
            writer_datas.append(manager.get_writer_data())
        agent_name_list = [manager.agent_name for manager in agent_manager_list]
    # preprocess agent stats
    data_list = []

    for id_agent, agent_name in enumerate(agent_name_list):
        writer_data = writer_datas[id_agent]
        if writer_data is not None:
            for idx in writer_data:
                if tag is None:
                    tags = list(writer_data[idx]["tag"].unique())
                    preprocess_funcs = [lambda x: x for _ in range(len(tags))]
                for id_tag, tag in enumerate(tags):
                    df = writer_data[idx]
                    processed_df = pd.DataFrame(df[df["tag"] == tag])
                    processed_df["value"] = preprocess_funcs[id_tag](
                        processed_df["value"].values
                    )
                    # update name according to AgentManager name and
                    # n_simulation
                    processed_df["name"] = agent_name
                    processed_df["n_simu"] = idx

                    processed_df = pd.concat(
                        [processed_df, df[df["tag"] != tag]], ignore_index=True
                    )
                    # add column
                    data_list.append(processed_df)
    all_writer_data = pd.concat(data_list, ignore_index=True)
    return all_writer_data


def _get_last_xp(input_dir, name):

    dir_name = Path(input_dir) / "manager_data"

    # list all of the experiments for this particular agent
    agent_xp = list(dir_name.glob(name + "*"))

    # get the times at which the experiment have been made
    times = [str(p).split("_")[-2] for p in agent_xp]
    days = [str(p).split("_")[-3] for p in agent_xp]
    hashs = [str(p).split("_")[-1] for p in agent_xp]
    datetimes = [
        datetime.strptime(days[i] + "_" + times[i], "%Y-%m-%d_%H-%M-%S")
        for i in range(len(days))
    ]

    if len(datetimes) == 0:
        raise ValueError(
            "input dir not found, verify that the agent are trained "
            'and that AgentManager.outdir_id_style="timestamp"'
        )

    # get the date of last experiment
    max_date = np.max(datetimes)
    id_max = np.argmax(datetimes)
    hash = hashs[id_max]
    agent_folder = (
        name + "_" + datetime.strftime(max_date, "%Y-%m-%d_%H-%M-%S") + "_" + hash
    )
    return agent_folder, dir_name


def is_bz_file(filepath):
    with open(filepath, "rb") as test_f:
        return test_f.read(2) == b"BZ"


def _load_data(agent_folder, dir_name, id_agent):
    writer_data = {}

    agent_dir = Path(dir_name) / agent_folder
    # list all the fits of this experiment
    exp_files = (agent_dir / Path("agent_handlers")).iterdir()
    nfit = len(list(exp_files))
    if nfit == 0:
        raise ValueError("Folders do not contain pickle files")

    if id_agent is not None:
        id_fits = [id_agent]
    else:
        id_fits = range(nfit)

    for ii in id_fits:
        # For each fit, load the writer data
        handler_name = agent_dir / Path(f"agent_handlers/idx_{ii}.pickle")
        try:
            if is_bz_file(handler_name):
                with bz2.BZ2File(handler_name, "rb") as ff:
                    tmp_dict = cPickle.load(ff)
            else:
                with handler_name.open("rb") as ff:
                    tmp_dict = pickle.load(ff)
        except Exception:
            if not is_bz_file(handler_name):
                with handler_name.open("rb") as ff:
                    tmp_dict = dill.load(ff)
            else:
                with bz2.BZ2File(handler_name, "rb") as ff:
                    tmp_dict = dill.load(ff)
        writer_data[str(ii)] = tmp_dict.get("_writer").data

    return writer_data


def plot_writer_data(
    data_source,
    tag,
    xtag=None,
    id_agent=None,
    ax=None,
    show=True,
    preprocess_func=None,
    title=None,
    savefig_fname=None,
    sns_kwargs=None,
    smooth_weight=0.09,
    plot_raw_curves=True,
):
    """
    Given a list of AgentManager or a folder, plot data (corresponding to info) obtained in each episode.
    The dictionary returned by agents' .fit() method must contain a key equal to `info`.

    If there are several simulations, a confidence interval is plotted ( 90% percentile interval if seaborn version >= 0.12.0
    otherwise standard deviation is used). This can be overridden using `sns_kwargs`.
    If there is only one simulation, a tensorboard-style smoothing is performed.

    Parameters
    ----------
    data_source : :class:`~rlberry.manager.AgentManager`, or list of :class:`~rlberry.manager.AgentManager` or str or list of str
        - If AgentManager or list of AgentManager, load data from it (the agents must be fitted).

        - If str, the string must be the string path of a directory,  each
        subdirectory of this directory must contain pickle files.
        load the data from the directory of the latest experiment in date.
        This str should be equal to the value of the `output_dir` parameter in
        :class:`~rlberry.manager.AgentManager`.

        - If list of str, each string must be a directory containing pickle files
        load the data from these pickle files.

        Note: the agent's save function must save its writer at the key `_writer`.
        This is the default for rlberry agents.
    tag : str
        Tag of data to plot on y-axis.
    xtag : str or None, default=None
        Tag of data to plot on x-axis. If None, use 'global_step' and aggregate data.
        If not None, only the data for one agents are used (i.e. from one fit,
        not from all n_fit fits). See id_agent in this case.
    id_agent : int or None, default=None
        id of the agent to plot, used only if xtag is not None.
    ax: matplotlib axis or None, default=None
        Matplotlib axis on which we plot. If None, create one. Can be used to
        customize the plot.
    show: bool, default=True
        If true, calls plt.show().
    preprocess_func: Callable, default=None
        Function to apply to 'tag' column before plot. For instance, if tag=episode_rewards,
        setting preprocess_func=np.cumsum will plot cumulative rewards. If None, do nothing.
    title: str (Optional)
        Optional title to plot. If None, set to tag.
    savefig_fname: str (Optional)
        Name of the figure in which the plot is saved with figure.savefig. If None,
        the figure is not saved.
    sns_kwargs: dict
        Optional extra params for seaborn lineplot.
    smooth_weight: float in [0,1], default=0.9
        Apply smoothing tensorboard-style to the plots with parameter smooth_weight.
        Only applied if there is one simulation only.
    plot_raw_curves: bool, default=True
        If True and if the curves are smoothed, plot also the unsmoothed curves.

    Returns
    -------
    Pandas DataFrame with processed data used by seaborn's lineplot.

    Examples
    --------
    >>> from rlberry.agents.torch import A2CAgent, DQNAgent
    >>> from rlberry.manager import AgentManager, plot_writer_data
    >>> from rlberry.envs import gym_make
    >>>
    >>> if __name__=="__main__":
    >>>     managers = [ AgentManager(
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
    >>>     data = plot_writer_data(managers, "episode_rewards", smooth_weight=0.95)
    """
    sns_kwargs = sns_kwargs or {}

    title = title or tag
    if preprocess_func is not None:
        ylabel = "value"
    else:
        ylabel = tag
    if (xtag is not None) and (id_agent is None):
        id_agent = 0
    processed_df = read_writer_data(
        data_source, tag, preprocess_func, id_agent=id_agent
    )

    if len(processed_df) == 0:
        logger.error("[plot_writer_data]: No data to be plotted.")
        return

    data = processed_df[processed_df["tag"] == tag]

    if xtag is None:
        xtag = "global_step"

    if data[xtag].notnull().sum() > 0:
        xx = xtag
        if data["global_step"].isna().sum() > 0:
            logger.warning(
                f"Plotting {tag} vs {xtag}, but {xtag} might be missing for some agents."
            )
    else:
        xx = data.index

    if ax is None:
        figure, ax = plt.subplots(1, 1)

    # Customizing linestyle in sns.lineplot
    if "style" in sns_kwargs:
        # Number of unique dash styles. Default: 4 styles max.
        n_uniq_dash = sns_kwargs.get("n_uniq_dash", 4)
        linestyles = ["", (1, 1), (5, 5), (1, 5, 3, 5)]
        # Cycle through default sns linestyles.
        dash_cycler = cycle(linestyles[:n_uniq_dash])
        sns_kwargs["dashes"] = [
            next(dash_cycler) for _ in range(data["name"].unique().size)
        ]

    # PS: in the next release of seaborn, ci should be deprecated and replaced
    # with errorbar, which allows to specifies other types of confidence bars,
    # in particular quantiles.

    def _smooth(df):
        df["value"] = smooth(list(df["value"]), smooth_weight)
        return df

    if len(data["n_simu"].unique()) == 1:
        legends = []
        if plot_raw_curves:
            sns.lineplot(x=xx, y="value", hue="name", data=data, ax=ax, alpha=0.3)
            legends += ["raw " + str(n) for n in data["name"].unique()]
        data = data.groupby(["name"]).apply(_smooth)
        lineplot_kwargs = dict(x=xx, y="value", hue="name", data=data, ax=ax)
        lineplot_kwargs.update(sns_kwargs)
        sns.lineplot(**lineplot_kwargs)
        ax.legend(legends + ["smoothed " + str(n) for n in data["name"].unique()])
    else:
        if LooseVersion(sns.__version__) >= LooseVersion("0.12.0"):
            if "errorbar" in sns_kwargs.keys():
                lineplot_kwargs = dict(x=xx, y="value", hue="name", data=data, ax=ax)
            else:
                # See https://seaborn.pydata.org/tutorial/error_bars
                lineplot_kwargs = dict(
                    x=xx, y="value", hue="name", data=data, ax=ax, errorbar=("pi", 90)
                )
        else:
            if "ci" in sns_kwargs.keys():
                lineplot_kwargs = dict(x=xx, y="value", hue="name", data=data, ax=ax)
            else:
                lineplot_kwargs = dict(
                    x=xx, y="value", hue="name", data=data, ax=ax, ci="sd"
                )
        lineplot_kwargs.update(sns_kwargs)
        sns.lineplot(**lineplot_kwargs)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    if show:
        plt.show()

    if savefig_fname is not None:
        plt.gcf().savefig(savefig_fname)

    return data


# https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar#_=_
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed
