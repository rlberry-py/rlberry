import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pickle
from rlberry.manager import AgentManager


logger = logging.getLogger(__name__)


def evaluate_agents(
    agent_manager_list,
    n_simulations=5,
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
    """
    sns_kwargs = sns_kwargs or {}

    #
    # evaluation
    #

    eval_outputs = []
    for agent_manager in agent_manager_list:
        logger.info(f"Evaluating {agent_manager.agent_name}...")
        outputs = agent_manager.eval_agents(n_simulations)
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


def read_writer_data(input_obj, tag, preprocess_func=None):
    """
    Given a list of AgentManager or a folder, read data (corresponding to info) obtained in each episode.
    The dictionary returned by agents' .fit() method must contain a key equal to `info`.

    Parameters
    ----------
    input_obj : AgentManager, or list of AgentManager or directory or list of directories
        If AgentManager or list of AgentManager, load data from it (the agents must be fitted).

        If directory, load the data from the directory of the latest experiment in date.
        If list of directories, load the data from these directories.

        Note: the agent's save function must save its writer at the key `_writer`.
        This is the default for rlberry agents.

    tag :  str or list of str
        Tag of data that we want to preprocess.

    preprocess_func: Callable or None or list of Callable or None
        Function to apply to 'tag' column before returning data.
        For instance, if tag=episode_rewards,setting preprocess_func=np.cumsum
        will return cumulative rewards
        If None, do not preprocess.
        If tag is a list, preprocess_func must be None or a list of Callable or
        None that matches the length of tag.


    Returns
    -------
    Pandas DataFrame with data from writers.
    """
    input_dir = None

    if not isinstance(input_obj, list):
        if isinstance(input_obj, AgentManager):
            input_obj = [input_obj]
        else:
            take_last_date = True
    else:
        take_last_date = False
        for dir in input_obj:
            files = list(Path(dir).iterdir())
            if len(files) == 0:
                raise RuntimeError(
                    "One of the files in input_obj does not contain pickle files"
                )

    if isinstance(input_obj[0], AgentManager):
        agent_manager_list = input_obj
    else:
        input_dir = input_obj

    if isinstance(tag, str):
        tags = [tag]
        preprocess_funcs = [preprocess_func or (lambda x: x)]
    else:
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
                writer_datas.append(_load_data(filename, dir_name))
        else:
            agent_name_list = [str(Path(p).stem).split("_")[0] for p in input_dir]
            agent_dirs = [str(Path(p).parent).split("_")[0] for p in input_dir]

            for id_f, filename in enumerate(input_dir):
                writer_datas.append(_load_data(filename, agent_dirs[id_f]))
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


def _load_data(agent_folder, dir_name):
    writer_data = {}

    agent_dir = Path(dir_name) / agent_folder
    # list all the fits of this experiment
    exp_files = (agent_dir / Path("agent_handlers")).iterdir()
    nfit = len(list(exp_files))
    if nfit == 0:
        raise ValueError("Folders do not contain pickle files")

    for ii in range(nfit):
        # For each fit, load the writer data
        handler_name = agent_dir / Path(f"agent_handlers/idx_{ii}.pickle")
        with handler_name.open("rb") as ff:
            tmp_dict = pickle.load(ff)
            writer_data[str(ii)] = tmp_dict.get("_writer").data
    return writer_data


def plot_writer_data(
    input_obj,
    tag,
    xtag=None,
    ax=None,
    show=True,
    preprocess_func=None,
    title=None,
    savefig_fname=None,
    sns_kwargs=None,
):
    """
    Given a list of AgentManager or a folder, plot data (corresponding to info) obtained in each episode.
    The dictionary returned by agents' .fit() method must contain a key equal to `info`.

    Parameters
    ----------
    input_obj : AgentManager, or list of AgentManager or directory or list of directories
        If AgentManager or list of AgentManager, load data from it (the agents must be fitted).

        If directory, load the data from the directory of the latest experiment in date.
        If list of directories, load the data from these directories.

        Note: the agent's save function must save its writer at the key `_writer`.
        This is the default for rlberry agents.
    tag : str
        Tag of data to plot on y-axis.
    xtag : str
        Tag of data to plot on x-axis. If None, use 'global_step'.
    ax: matplotlib axis
        Matplotlib axis on which we plot. If None, create one. Can be used to
        customize the plot.
    show: bool
        If true, calls plt.show().
    preprocess_func: Callable
        Function to apply to 'tag' column before plot. For instance, if tag=episode_rewards,
        setting preprocess_func=np.cumsum will plot cumulative rewards
    title: str (Optional)
        Optional title to plot. If None, set to tag.
    savefig_fname: str (Optional)
        Name of the figure in which the plot is saved with figure.savefig. If None,
        the figure is not saved.
    sns_kwargs: dict
        Optional extra params for seaborn lineplot.

    Returns
    -------
    Pandas DataFrame with processed data used by seaborn's lineplot.
    """
    sns_kwargs = sns_kwargs or {}

    title = title or tag
    if preprocess_func is not None:
        ylabel = "value"
    else:
        ylabel = tag
    processed_df = read_writer_data(input_obj, tag, preprocess_func)
    # add column with xtag, if given
    if xtag is not None:
        df_xtag = pd.DataFrame(processed_df[processed_df["tag"] == xtag])
        processed_df[xtag] = df_xtag["value"].values
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

    # PS: in the next release of seaborn, ci should be deprecated and replaced
    # with errorbar, which allows to specifies other types of confidence bars,
    # in particular quantiles.
    lineplot_kwargs = dict(
        x=xx, y="value", hue="name", style="name", data=data, ax=ax, ci="sd"
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
