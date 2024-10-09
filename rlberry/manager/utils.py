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


def tensorboard_folder_to_dataframe_for_plotting(path_to_tensorboard_data):
    """
    path_to_tensorboard_data : path to the tensorboard data. It must be the parent folder of all the training, then respect the architecture : <path_to_tensorboard_data/algo_name/n_simu/events.out.xxxxx>

    Return a dict of panda dataframe (key = tag, value = panda.dataframe)
    """
    from tensorflow.python.summary.summary_iterator import summary_iterator

    dataframe_by_tag = {}
    for algo_name in os.listdir(path_to_tensorboard_data):
        path_for_this_algo = os.path.join(path_to_tensorboard_data, algo_name)
        if os.path.isdir(path_for_this_algo):
            for seed in os.listdir(path_for_this_algo):
                current_seed_path = os.path.join(path_for_this_algo, seed)
                content = os.listdir(current_seed_path)
                assert len(content) == 1  # should be "events.out.tfevents.xxxxxxxxx"
                content_path = os.path.join(current_seed_path, content[0])

                for summary in summary_iterator(content_path):
                    if summary.source_metadata.writer:  # skip useless iteration
                        continue

                    dict_data = _summary_value_to_dict(summary.summary.value)
                    current_tag = dict_data["tag"]

                    if (
                        current_tag not in dataframe_by_tag
                    ):  # new tag, create new entry in the dict
                        dataframe_by_tag[current_tag] = []

                    value = dict_data["simple_value"]
                    dataframe_by_tag[current_tag].append(
                        [algo_name, seed, summary.step, value]
                    )

    # convert the "dict of array" to "dict of panda dataframe"
    df = {}
    for tag, value in dataframe_by_tag.items():
        df[tag] = pd.DataFrame(value, columns=["name", "n_simu", "x", "y"])
    return df


def _summary_value_to_dict(value_to_convert):
    """
    convert something like :
    -------
    "[tag: "rollout/ep_len_mean"
    simple_value: 25.15
    ]"
    -------
    to dict :  {'tag': 'rollout/ep_len_mean', 'simple_value': 25.15}
    """
    result_dict = {}

    # Removing [] and splitting into lines
    lines = str(value_to_convert).strip().strip("[]").splitlines()

    for line in lines:
        # Remove blanks at the beginning and end of lines
        line = line.strip()

        if line:  # If the line is not empty
            key, value = line.split(": ", 1)  # Separate key and value
            value = value.strip('"')  # Remove quotes

            # Convert to float if possible, otherwise keep as string
            if value.replace(".", "", 1).isdigit():
                value = float(value)

            result_dict[key] = value

    return result_dict
