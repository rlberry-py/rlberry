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


def tensorboard_folder_to_dataframe(path_to_tensorboard_data):
    """
    path_to_tensorboard_data : path to the tensorboard data. It must be the parent folder of all the training, and the event have to be in this kind of path : <path_to_tensorboard_data/algo_name/n_simu/events.out.tfevents.xxxxx>

    Return a dict of panda dataframe (key = tag, value = panda.dataframe)
    """
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

    # convert the "dict of array" to "dict of panda dataframe"
    df = {}
    for tag, value in dataframe_by_tag.items():
        df[tag] = pd.DataFrame(value, columns=["name", "n_simu", "x", "y"])
    return df
