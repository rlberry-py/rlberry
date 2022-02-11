import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pickle
from copy import deepcopy

from rlberry import metadata_utils


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


def plot_writer_data(
    agent_manager,
    tag,
    xtag=None,
    fignum=None,
    show=True,
    preprocess_func=None,
    title=None,
    sns_kwargs=None,
):
    """
    Given a list of AgentManager, plot data (corresponding to info) obtained in each episode.
    The dictionary returned by agents' .fit() method must contain a key equal to `info`.

    Parameters
    ----------
    agent_manager : AgentManager, or list of AgentManager
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
    sns_kwargs: dict
        Optional extra params for seaborn lineplot.

    Returns
    -------
    Pandas DataFrame with processed data used by seaborn's lineplot.
    """
    sns_kwargs = sns_kwargs or {"ci": "sd"}

    title = title or tag
    if preprocess_func is not None:
        ylabel = "value"
    else:
        ylabel = tag
    processed_df = read_writer_data(agent_manager, tag, preprocess_func)
    # add column with xtag, if given
    if xtag is not None:
        df_xtag = pd.DataFrame(processed_df[processed_df["tag"] == xtag])
        processed_df[xtag] = df_xtag["value"].values
    if len(processed_df) == 0:
        logger.error("[plot_writer_data]: No data to be plotted.")
        return

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

    lineplot_kwargs = dict(
        x=xx, y="value", hue="name", style="name", data=data, ax=ax, ci="sd"
    )
    lineplot_kwargs.update(sns_kwargs)
    sns.lineplot(**lineplot_kwargs)
    plt.title(title)
    plt.ylabel(ylabel)

    if show:
        plt.show()

    return data
