import os

import pathlib
import rlberry

logger = rlberry.logger


def get_single_path_of_most_recently_trained_experiment_manager_obj_from_path(
    path_to_explore,
):
    """
    Get the path to the latest manager_obj.pickle (whatever the agent) in a directory (sort them by reverse last modification date, and give the first).

    Parameters
    ----------
    path_to_explore str: the path where the ExperimentManager was saved... (the parent folder of the manager_obj.pickle)

    """
    paths = list(
        pathlib.Path(path_to_explore).glob("**" + os.sep + "manager_obj.pickle")
    )
    paths.sort(
        key=lambda x: os.path.getmtime(x), reverse=True
    )  # Sort based on file last modification time
    return paths[0]


def get_all_path_of_most_recently_trained_experiments_manager_obj_from_path(
    path_to_explore,
):
    """
    Get the path to all the more recently updated/created manager_obj.pickle (by agent) in a directory (sort them by reverse last modification date, and give the first).

    Parameters
    ----------
    path_to_explore str: the path where the ExperimentManager was saved... (the parent folder of the manager_obj.pickle)

    ----------
    return: list with the path of all the more recently updated/created manager_obj.pickle by different agent.

    """
    dict_of_all_exp = get_dict_of_all_experiment_manager_obj_from_path(path_to_explore)

    paths_to_return = []
    for key, value in dict_of_all_exp.items():
        sorted_values = sorted(value, key=lambda x: os.path.getmtime(x), reverse=True)
        paths_to_return.append(sorted_values[0])
    return paths_to_return


def get_dict_of_all_experiment_manager_obj_from_path(path_to_explore):
    """
    Find the paths of all the manager_obj.pickle contained in a directory

    Parameters
    ----------
    path_to_explore str: the path where the ExperimentManager were saved... (the parent folder of all the manager_obj.pickle)

    ----------
    return: Dict with the path of all the manager_obj.pickle

    Dict format :
    key = Agent name
    value = [(path1,date1),(path2,date2),...]
    """

    # get all the "manager_obj.pickle" from the directory
    all_paths = list(
        pathlib.Path(path_to_explore).glob("**" + os.sep + "manager_obj.pickle")
    )

    # isolate the name of all the agent of directory
    list_agent_name = []
    for path in all_paths:
        half_split = str(path).split(os.sep + "manager_data" + os.sep)[-1]
        list_agent_name.append(half_split.split("_")[0])
    list_unique_agent_name = list(set(list_agent_name))

    # for each of this agent, get the latest manager_obj.pickle (by finding all, and keep the first after sorting by last modification time).
    dict_to_return = {}
    for unique_name in list_unique_agent_name:
        regex_to_search = (
            "**"
            + os.sep
            + "manager_data"
            + os.sep
            + ""
            + unique_name
            + "_*"
            + os.sep
            + "manager_obj.pickle"
        )
        dict_to_return[unique_name] = list(
            pathlib.Path(path_to_explore).glob(regex_to_search)
        )

    return dict_to_return
