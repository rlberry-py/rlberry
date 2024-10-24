import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd

# from pathlib import Path
# from datetime import datetime
# import pickle
# import bz2
# import _pickle as cPickle
# import dill

from rlberry.manager import ExperimentManager
from rlberry.utils import loading_tools
import rlberry

logger = rlberry.logger


def evaluate_agents(
    experiment_manager_list,
    n_simulations=5,
    choose_random_agents=True,
    fignum=None,
    show=True,
    plot=True,
):
    """
    Evaluate and compare each of the agents in experiment_manager_list.

    Parameters
    ----------
    experiment_manager_list: list of ExperimentManager objects.
    n_simulations: int
        Number of calls to the eval() method of each ExperimentManager instance.
    choose_random_agents: bool
        If true and n_fit>1, use a random fitted agent from each ExperimentManager at each evaluation.
        Otherwise, each fitted agent of each ExperimentManager is evaluated n_simulations times.
    fignum: string or int
        Identifier of plot figure.
    show: bool
        If true, calls plt.show().
    plot: bool
        If false, do not plot.

    Returns
    -------
    dataframe with the evaluation results.

    Examples
    --------
    >>> from rlberry.agents.stable_baselines import StableBaselinesAgent
    >>> from stable_baselines3 import PPO, A2C
    >>> from rlberry.manager import ExperimentManager, evaluate_agents
    >>> from rlberry.envs import gym_make
    >>> import matplotlib.pyplot as plt
    >>>
    >>> if __name__=="__main__":
    >>>     names = ["A2C", "PPO"]
    >>>     managers = [ ExperimentManager(
    >>>         StableBaselinesAgent,
    >>>         (gym_make, dict(id="Acrobot-v1")),
    >>>         fit_budget=1e5,
    >>>         agent_name = names[i],
    >>>         eval_kwargs=dict(eval_horizon=500),
    >>>         init_kwargs= {"algo_cls": algo_cls, "policy": "MlpPolicy", "verbose": 0},
    >>>         n_fit=1,
    >>>         seed=42,
    >>>          ) for i, algo_cls in enumerate([A2C, PPO])]
    >>>     for manager in managers:
    >>>         manager.fit()
    >>>     data = evaluate_agents(managers, n_simulations=50)

    """

    #
    # evaluation
    #

    eval_outputs = []
    for experiment_manager in experiment_manager_list:
        logger.info(f"Evaluating {experiment_manager.agent_name}...")
        if choose_random_agents:
            outputs = experiment_manager.eval_agents(n_simulations)
        else:
            outputs = []
            for idx in range(len(experiment_manager.agent_handlers)):
                outputs += list(
                    experiment_manager.eval_agents(n_simulations, agent_id=idx)
                )

        if len(outputs) > 0:
            eval_outputs.append(outputs)

    if len(eval_outputs) == 0:
        logger.error(
            "[evaluate_agents]: No evaluation data. Make sure ExperimentManager.fit() has been called."
        )
        return

    #
    # plot
    #

    # build unique agent IDs (in case there are two agents with the same ID)
    unique_ids = []
    id_count = {}
    for experiment_manager in experiment_manager_list:
        name = experiment_manager.agent_name
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
        plt.boxplot(output.values, tick_labels=output.columns)
        plt.xlabel("agent")
        plt.ylabel("evaluation output")
        if show:
            plt.show()

    return output


def _preprocess_list_datasource_str(data_source, many_agent_by_str_datasource):
    """
    Get a list of ExperimentManager from a data_source that is a list of 'str'

    Parameters
    ----------
    data_source:
        check 'read_writer_data' doc below
    many_agent_by_str_datasource: bool (False by default)
        check 'read_writer_data' doc below
    """
    experiment_manager_list = []
    if many_agent_by_str_datasource:
        for path in data_source:
            paths_to_load = loading_tools.get_all_path_of_most_recently_trained_experiments_manager_obj_from_path(
                path
            )
            for current_path in paths_to_load:
                experiment_manager_list.append(ExperimentManager.load(current_path))
    else:
        for path in data_source:
            # find the path to the latest "manager_obj.pickle"
            path_to_load = loading_tools.get_single_path_of_most_recently_trained_experiment_manager_obj_from_path(
                path
            )
            experiment_manager_list.append(ExperimentManager.load(path_to_load))
    return experiment_manager_list


def _preprocess_not_a_list_datasource_str(data_source, many_agent_by_str_datasource):
    """
    Get a list of ExperimentManager from a data_source that is a 'str'

    Parameters
    ----------
    data_source:
        check 'read_writer_data' doc below
    many_agent_by_str_datasource: bool (False by default)
        check 'read_writer_data' doc below
    """
    if many_agent_by_str_datasource:
        paths_to_load = loading_tools.get_all_path_of_most_recently_trained_experiments_manager_obj_from_path(
            data_source
        )
        experiment_manager_list = []
        for current_path in paths_to_load:
            experiment_manager_list.append(ExperimentManager.load(current_path))
    else:
        # find the path to the latest "manager_obj.pickle"
        paths_to_load = loading_tools.get_single_path_of_most_recently_trained_experiment_manager_obj_from_path(
            data_source
        )
        experiment_manager_list = [ExperimentManager.load(paths_to_load)]
    return experiment_manager_list


def _preprocess_list_datasource(data_source, many_agent_by_str_datasource):
    """
    Get a list of ExperimentManager from a data_source that is a list (so a list of 'str' or a list of 'ExperimentManager')

    Parameters
    ----------
    data_source:
        check 'read_writer_data' doc below
    many_agent_by_str_datasource: bool (False by default)
        check 'read_writer_data' doc below
    """
    if isinstance(data_source[0], ExperimentManager):
        experiment_manager_list = data_source
    else:
        experiment_manager_list = _preprocess_list_datasource_str(
            data_source, many_agent_by_str_datasource
        )
    return experiment_manager_list


def _preprocess_not_a_list_datasource(data_source, many_agent_by_str_datasource):
    """
    Get a list of ExperimentManager from a data_source that is not a list (so a 'str' or an 'ExperimentManager')

    Parameters
    ----------
    data_source:
        check 'read_writer_data' doc below
    many_agent_by_str_datasource: bool (False by default)
        check 'read_writer_data' doc below
    """
    if isinstance(data_source, ExperimentManager):
        experiment_manager_list = [data_source]
    else:
        experiment_manager_list = _preprocess_not_a_list_datasource_str(
            data_source, many_agent_by_str_datasource
        )
    return experiment_manager_list


def read_writer_data(
    data_source,
    many_agent_by_str_datasource=False,
    preprocess_tag=None,
    preprocess_func=None,
    id_agent=None,
):
    """
    Given a list of ExperimentManager or a folder, read data (corresponding to info) obtained in each episode.
    The dictionary returned by agents' .fit() method must contain a key equal to `info`.

    Parameters
    ----------
    data_source: :class:`~rlberry.manager.ExperimentManager`, or list of :class:`~rlberry.manager.ExperimentManager` or str or list of str
        - If ExperimentManager or list of ExperimentManager, load data from it (the agents must be fitted).

        - If str, the string must be the string path of a directory,  each
        subdirectory of this directory must contain pickle files.
        Load the data from the directory of the latest experiment in date.
        This str should be equal to the value of the `output_dir` parameter in
        :class:`~rlberry.manager.ExperimentManager`.

        - If list of str, each string must be a directory containing pickle files.
        Load the data from these pickle files.

        Note: the agent's save function must save its writer at the key `_writer`.
        This is the default for rlberry agents.

    many_agent_by_str_datasource: bool (False by default)
        (only useful if the datasource is a str or a list of str)
        If True, select many agent: the last agent of every different agent name inside the folder
        If False, select only one agent: the most recently trained agent from the directory (whatever its name)

    preprocess_tag:  str or list of str or None
        Tag of data that we want to preprocess.

    preprocess_func: Callable or None or list of Callable or None
        Function to apply to 'preprocess_tag' column before returning data.
        For instance, if preprocess_tag=episode_rewards,setting preprocess_func=np.cumsum
        will return cumulative rewards
        If None, do not preprocess.
        If preprocess_tag is a list, preprocess_func must be None or a list of Callable or
        None that matches the length of preprocess_tag.

    id_agent: int or None, default=None
        If not None, returns the data only for agent 'id_agent'.

    Returns
    -------
    Pandas DataFrame with data from writers.

    Examples
    --------
    >>> from rlberry.agents.torch import A2CAgent, DQNAgent
    >>> from rlberry.manager import ExperimentManager, read_writer_data
    >>> from rlberry.envs import gym_make
    >>>
    >>> if __name__=="__main__":
    >>>     managers = [ ExperimentManager(
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

    if isinstance(data_source, list):
        experiment_manager_list = _preprocess_list_datasource(
            data_source, many_agent_by_str_datasource
        )
    else:
        experiment_manager_list = _preprocess_not_a_list_datasource(
            data_source, many_agent_by_str_datasource
        )

    if isinstance(preprocess_tag, str):
        tags = [preprocess_tag]
        preprocess_funcs = [preprocess_func or (lambda x: x)]
    elif isinstance(preprocess_tag, list):
        tags = preprocess_tag
        if preprocess_func is None:
            preprocess_funcs = [lambda x: x for _ in range(len(tags))]
        else:
            assert len(preprocess_func) == len(tags)
            preprocess_funcs = preprocess_func
    else:
        tags = None

    writer_datas = []
    for manager in experiment_manager_list:
        writer_datas.append(manager.get_writer_data())
    agent_name_list = [manager.agent_name for manager in experiment_manager_list]
    # preprocess agent stats
    data_list = []

    for id_agent, agent_name in enumerate(agent_name_list):
        writer_data = writer_datas[id_agent]
        if writer_data is not None:
            for idx in writer_data:
                df = writer_data[idx]
                if tags:
                    # add the column that don't need to be preprocessed
                    processed_df = pd.DataFrame(df[~df["tag"].isin(tags)])

                    # add the preprocessed column
                    for id_tag, tag in enumerate(tags):
                        new_column = pd.DataFrame(df[df["tag"] == tag])
                        new_column["value"] = preprocess_funcs[id_tag](
                            new_column["value"].values
                        )
                        processed_df = pd.concat(
                            [processed_df, new_column], ignore_index=True
                        )
                else:
                    processed_df = pd.DataFrame(df)

                # update name according to ExperimentManager name and n_simulation
                processed_df["name"] = agent_name
                processed_df["n_simu"] = idx
                data_list.append(processed_df)

    all_writer_data = pd.concat(data_list, ignore_index=True)
    return all_writer_data


######## old code to remove ?
# def _get_last_xp(input_dir, name):
#     dir_name = Path(input_dir) / "manager_data"

#     # list all of the experiments for this particular agent
#     agent_xp = list(dir_name.glob(name + "*"))

#     # get the times at which the experiment have been made
#     times = [str(p).split("_")[-2] for p in agent_xp]
#     days = [str(p).split("_")[-3] for p in agent_xp]
#     hashs = [str(p).split("_")[-1] for p in agent_xp]
#     datetimes = [
#         datetime.strptime(days[i] + "_" + times[i], "%Y-%m-%d_%H-%M-%S")
#         for i in range(len(days))
#     ]

#     if len(datetimes) == 0:
#         raise ValueError(
#             "input dir not found, verify that the agent are trained "
#             'and that ExperimentManager.outdir_id_style="timestamp"'
#         )

#     # get the date of last experiment
#     max_date = np.max(datetimes)
#     id_max = np.argmax(datetimes)
#     hash = hashs[id_max]
#     agent_folder = (
#         name + "_" + datetime.strftime(max_date, "%Y-%m-%d_%H-%M-%S") + "_" + hash
#     )
#     return agent_folder, dir_name


# ######## old code to remove ?
# def is_bz_file(filepath):
#     with open(filepath, "rb") as test_f:
#         return test_f.read(2) == b"BZ"


######## old code to remove ?
# def _load_data(agent_folder, dir_name, id_agent):
#     writer_data = {}

#     agent_dir = Path(dir_name) / agent_folder
#     # list all the fits of this experiment
#     exp_files = (agent_dir / Path("agent_handlers")).iterdir()
#     nfit = len(
#         [1 for a_ in [str(e).split(".") for e in exp_files] if a_[-1] == "pickle"]
#     )
#     # nfit = len(list(exp_files))
#     if nfit == 0:
#         raise ValueError("Folders do not contain pickle files")

#     if id_agent is not None:
#         id_fits = [id_agent]
#     else:
#         id_fits = range(nfit)

#     for ii in id_fits:
#         # For each fit, load the writer data
#         handler_name = agent_dir / Path(f"agent_handlers/idx_{ii}.pickle")
#         try:
#             if is_bz_file(handler_name):
#                 with bz2.BZ2File(handler_name, "rb") as ff:
#                     tmp_dict = cPickle.load(ff)
#             else:
#                 with handler_name.open("rb") as ff:
#                     tmp_dict = pickle.load(ff)
#         except Exception:
#             if not is_bz_file(handler_name):
#                 with handler_name.open("rb") as ff:
#                     tmp_dict = dill.load(ff)
#             else:
#                 with bz2.BZ2File(handler_name, "rb") as ff:
#                     tmp_dict = dill.load(ff)
#         writer_data[str(ii)] = tmp_dict.get("_writer").data

#     return writer_data
