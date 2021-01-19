from pathlib import Path
from rlberry.stats import AgentStats
import pandas as pd
import logging


logger = logging.getLogger(__name__)


def _get_most_recent_path(path_list):
    """
    get most recend result for each agent_name
    """
    most_recent_path = None
    most_recent_id = None
    for ii, dd in enumerate(path_list):
        try:
            run_id = int(dd.name)
            if ii == 0 or run_id > most_recent_id:
                most_recent_path = dd
                most_recent_id = run_id
        except Exception:
            continue
    return most_recent_path


def load_experiment_results(output_dir, experiment_name):
    """
    Parameters
    ----------
    output_dir : str or Path
        directory where experiment results are stored
        (command line argument --output_dir when running the eperiment)
    experiment_name : str or Path
        name of yaml file describing the experiment.

    Returns
    -------
    output_data: dict
        dictionary such that :code:`output_data[agent_name] = (agent_stats, dataframes)`,
        a tuple containing an AgentStats instance and a dict of pandas data frames
        from the last run of an experiment.
    """
    results_dir = Path(output_dir) / Path(experiment_name).stem
    # Subdirectories with data for each agent
    subdirs = [f for f in results_dir.iterdir() if f.is_dir()]

    # Create dictionary dict[agent_name] = most recent result dir
    data_dirs = {}
    for dd in subdirs:
        data_dirs[dd.name] = _get_most_recent_path([f for f in dd.iterdir() if f.is_dir()])

    # Load data from each subdir
    output_data = {}
    for agent_name in data_dirs:
        fname = data_dirs[agent_name] / 'stats.pickle'
        try:
            stats = AgentStats.load(fname)
        except Exception:
            pass

        logger.info("... loaded " + str(fname))

        dataframes = {}
        csv_files = [f for f in data_dirs[agent_name].iterdir() if f.suffix == '.csv']
        for ff in csv_files:
            dataframes[ff.stem] = pd.read_csv(ff)
            logger.info("... loaded " + str(ff))

        output_data[agent_name] = (stats, dataframes)

    return output_data
