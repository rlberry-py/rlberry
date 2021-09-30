from pathlib import Path
from rlberry.manager import AgentManager
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _get_most_recent_path(path_list):
    """
    Get most recent result for each agent_name.
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
    output_dir : str or Path, or list
        directory (or list of directories) where experiment results are stored
        (command line argument --output_dir when running the eperiment)
    experiment_name : str or Path, or list
        name of yaml file describing the experiment.

    Returns
    -------
    output_data: dict
        dictionary such that

        output_data['experiment_dirs'] = list of paths to experiment directory (output_dir/experiment_name)
        output_data['agent_list'] = list containing the names of the agents in the experiment
        output_data['manager'][agent_name] = fitted AgentManager for agent_name
        output_data['dataframes'][agent_name] = dict of pandas data frames from the last run of the experiment
        output_data['data_dir'][agent_name] = directory from which the results were loaded
    """
    output_data = {}
    output_data['agent_list'] = []
    output_data['manager'] = {}
    output_data['dataframes'] = {}
    output_data['data_dir'] = {}

    # preprocess input
    if not isinstance(output_dir, list):
        output_dir = [output_dir]
    if not isinstance(experiment_name, list):
        experiment_name = [experiment_name]
    ndirs = len(output_dir)

    if ndirs > 1:
        assert len(experiment_name) == ndirs, "Number of experiment names must match the number of output_dirs "
    else:
        output_dir = len(experiment_name) * output_dir

    results_dirs = []
    for dd, exper in zip(output_dir, experiment_name):
        results_dirs.append(Path(dd) / Path(exper).stem)
    output_data['experiment_dirs'] = results_dirs

    # Subdirectories with data for each agent
    subdirs = []
    for dd in results_dirs:
        subdirs.extend([f for f in dd.iterdir() if f.is_dir()])

    # Create dictionary dict[agent_name] = most recent result dir
    data_dirs = {}
    for dd in subdirs:
        data_dirs[dd.name] = _get_most_recent_path([f for f in dd.iterdir() if f.is_dir()])
        data_dirs[dd.name] = data_dirs[dd.name] / 'manager_data'

    # Load data from each subdir
    for agent_name in data_dirs:
        output_data['agent_list'].append(agent_name)

        # store data_dir
        output_data['data_dir'][agent_name] = data_dirs[agent_name]

        # store AgentManager
        output_data['manager'][agent_name] = None
        fname = data_dirs[agent_name] / 'manager_obj.pickle'
        try:
            output_data['manager'][agent_name] = AgentManager.load(fname)
        except Exception:
            logger.warning(f'Could not load AgentManager instance for {agent_name}.')
        logger.info("... loaded " + str(fname))

        # store data frames
        dataframes = {}
        csv_files = [f for f in data_dirs[agent_name].iterdir() if f.suffix == '.csv']
        for ff in csv_files:
            dataframes[ff.stem] = pd.read_csv(ff)
            logger.info("... loaded " + str(ff))
        output_data['dataframes'][agent_name] = dataframes

    return output_data
