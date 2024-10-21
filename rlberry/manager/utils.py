import sqlite3
import os
import pandas as pd


def create_database(db_file):
    """Create a connection to a SQLite database."""
    connection = None
    try:
        connection = sqlite3.connect(db_file)
        print(f"Connected to {db_file} (sqlite3 version = {sqlite3.version})")
    except sqlite3.Error as err:
        print(err)

    if connection:
        connection.close()
        return True
    return False


def tensorboard_to_dataframe(tensorboard_data):
    """
    Function to convert 'tensorboard log' to 'Panda DataFrames'.

    | To convert the 'tensorboard log', the input must be must be the path to "the parent folder of all the training log" (path_to_tensorboard_data), and the 'events.out.tfevents' files have to be in this kind of path :
    < path_to_tensorboard_data/algo_name/n_simu/events.out.tfevents.xxxxx >

    Or you can specify all the desired 'events.out.tfevents' with a Dict. In that case, the key should be the algorithm name, and the value the list of the 'events.out.tfevents' path. The  seed/n_simu number wille be the position in the list.

    The output format is a dictionary.

    | key = tag (type of data)
    | value = Panda DataFrame with the following structure (4 column):

        * "name" = algo_name
        * "n_simu" = n_simu (seed)
        * "x" = step number
        * "y" = value of the data

    Parameters
    ----------
    path_to_tensorboard_data (str or Dict):
        if str: path to the parent folder of the tensorboard's data.
        if dict: Key = algo_name , value = list of 'events.out.tfevents.xxxxx' path


    Returns
    -------
    Dict : dict of Panda DataFrame (key = tag, value = Panda.DataFrame)
    """

    dataframe_by_tag = {}

    if isinstance(tensorboard_data, str):
        dataframe_by_tag = _tensorboard_to_dataframe_from_parent_path(tensorboard_data)
    elif isinstance(tensorboard_data, dict):
        dataframe_by_tag = _tensorboard_to_dataframe_from_dict_paths(tensorboard_data)
    else:
        raise IOError(
            str(
                "Input of 'tensorboard_to_dataframe' must be a str or a dict... not a "
                + str(type(tensorboard_data))
            )
        )

    # convert the "dict of array" to "dict of panda dataframe"
    df = {}
    for tag, value in dataframe_by_tag.items():
        df[tag] = pd.DataFrame(value, columns=["name", "n_simu", "x", "y"])
    return df


def _tensorboard_to_dataframe_from_parent_path(path_to_tensorboard_data):
    from tensorboard.backend.event_processing import event_accumulator

    dataframe_by_tag = {}
    for algo_name in os.listdir(path_to_tensorboard_data):
        path_for_this_algo = os.path.join(path_to_tensorboard_data, algo_name)
        if os.path.isdir(path_for_this_algo):
            for seed in os.listdir(path_for_this_algo):
                current_seed_path = os.path.join(path_for_this_algo, seed)
                content = os.listdir(current_seed_path)
                assert len(content) == 1  # should be "events.out.tfevents.xxxxxxxxx"
                content_path = os.path.join(current_seed_path, content[0])

                # load the event in the file, and get the tags
                ea = event_accumulator.EventAccumulator(content_path)
                ea.Reload()
                scalar_tags = ea.Tags()["scalars"]

                for tag in scalar_tags:
                    events = ea.Scalars(tag)
                    if (
                        tag not in dataframe_by_tag
                    ):  # new tag, create new entry in the dict
                        dataframe_by_tag[tag] = []
                    new_elements = [(algo_name, seed, e.step, e.value) for e in events]
                    dataframe_by_tag[tag].extend(new_elements)
    return dataframe_by_tag


def _tensorboard_to_dataframe_from_dict_paths(dict_tensorboard_data):
    from tensorboard.backend.event_processing import event_accumulator

    dataframe_by_tag = {}
    for algo_name, current_path_list in dict_tensorboard_data.items():
        for idx, path in enumerate(current_path_list):
            # load the event in the file, and get the tags
            ea = event_accumulator.EventAccumulator(path)
            ea.Reload()
            scalar_tags = ea.Tags()["scalars"]

            for tag in scalar_tags:
                events = ea.Scalars(tag)
                if tag not in dataframe_by_tag:  # new tag, create new entry in the dict
                    dataframe_by_tag[tag] = []
                new_elements = [(algo_name, idx, e.step, e.value) for e in events]
                dataframe_by_tag[tag].extend(new_elements)
    return dataframe_by_tag


# Faire un test qui vérifie la 2ème fonction, et un test qui vérifie qu'elles donnent les mêmes résultats
