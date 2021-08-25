import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)


def evaluate_agents(agent_stats_list,
                    n_simulations=5,
                    fignum=None,
                    show=True,
                    plot=True,
                    sns_kwargs=None):
    """
    Evaluate and compare each of the agents in agent_stats_list.

    Parameters
    ----------
    agent_stats_list : list of AgentStats objects.
    n_simulations: int
        Number of calls to the eval() method of each AgentStats instance.
    fignum: string or int
        Identifier of plot figure.
    show: bool
        If true, calls plt.show().
    plot: bool
        If false, do not plot.
    sns_kwargs:
        Extra parameters for sns.boxplot
    """
    sns_kwargs = sns_kwargs or {}

    #
    # evaluation
    #

    eval_outputs = []
    for agent_stats in agent_stats_list:
        outputs = []
        eval_env = agent_stats.build_eval_env()
        for _ in range(n_simulations):
            outputs.append(agent_stats.eval(eval_env))

        eval_outputs.append(outputs)
    #
    # plot
    #

    # build unique agent IDs (in case there are two agents with the same ID)
    unique_ids = []
    id_count = {}
    for agent_stats in agent_stats_list:
        name = agent_stats.agent_name
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


def plot_writer_data(agent_stats,
                     tag,
                     fignum=None,
                     show=True,
                     preprocess_func=None,
                     title=None,
                     sns_kwargs=None):
    """
    Given a list of AgentStats, plot data (corresponding to info) obtained in each episode.
    The dictionary returned by agents' .fit() method must contain a key equal to `info`.

    Parameters
    ----------
    agent_stats : AgentStats, or list of AgentStats
    tag : str
        Tag of data to plot.
    fignum: string or int
        Identifier of plot figure.
    show: bool
        If true, calls plt.show().
    preprocess_func: Callable
        Function to apply to 'tag' column before plot. For instance, if tag=episode_rewards,
        setting preprocess_func=np.cumsum will plot cumulative rewards
    title: str (Optional)
        Optional title to plot. If None, set to tag.
    sns_kwargs: dict
        Optional extra params for seaborn lineplot.
    """
    sns_kwargs = sns_kwargs or {'ci': 'sd'}

    plt.figure(fignum)
    title = title or tag
    if preprocess_func is not None:
        ylabel = 'value'
    else:
        ylabel = tag
    preprocess_func = preprocess_func or (lambda x: x)
    agent_stats_list = agent_stats
    if not isinstance(agent_stats_list, list):
        agent_stats_list = [agent_stats_list]

    # preprocess agent stats
    data_list = []
    for stats in agent_stats_list:
        if stats.writer_data is not None:
            for idx in stats.writer_data:
                df = stats.writer_data[idx]
                df = pd.DataFrame(df[df['tag'] == tag])
                df['value'] = preprocess_func(df['value'].values)
                data_list.append(df)

    all_writer_data = pd.concat(data_list, ignore_index=True)

    data = all_writer_data[all_writer_data['tag'] == tag]
    if data['global_step'].notnull().sum() > 0:
        xx = 'global_step'
        if data['global_step'].isna().sum() > 0:
            logger.warning(f'Plotting {tag} vs global_step, but global_step might be missing for some agents.')
    else:
        xx = data.index

    sns.lineplot(x=xx, y='value', hue='name', data=data, **sns_kwargs)
    plt.title(title)
    plt.ylabel(ylabel)

    if show:
        plt.show()
