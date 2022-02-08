"""Run experiments.

Usage:
    run.py <experiment_path> [--enable_tensorboard] [--n_fit=<nf>] [--output_dir=<dir>] [--parallelization=<par>] [--max_workers=<workers>]
    run.py (-h | --help)

Options:
    -h --help                Show this screen.
    --enable_tensorboard     Enable tensorboard writer in AgentManager.
    --n_fit=<nf>             Number of times each agent is fit [default: 4].
    --output_dir=<dir>       Directory to save the results [default: results].
    --parallelization=<par>  Either 'thread' or 'process' [default: process].
    --max_workers=<workers>  Number of workers used by AgentManager.fit. Set to -1 for the maximum value. [default: -1]
"""
import logging
from docopt import docopt
from pathlib import Path
from rlberry.experiment.yaml_utils import parse_experiment_config
from rlberry.manager import AgentManager
from rlberry import check_packages

logger = logging.getLogger(__name__)


def experiment_generator():
    """
    Parse command line arguments and yields AgentManager instances.
    """
    args = docopt(__doc__)
    max_workers = int(args["--max_workers"])
    if max_workers == -1:
        max_workers = None
    for (_, agent_manager_kwargs) in parse_experiment_config(
        Path(args["<experiment_path>"]),
        n_fit=int(args["--n_fit"]),
        max_workers=max_workers,
        output_base_dir=args["--output_dir"],
        parallelization=args["--parallelization"],
    ):
        if args["--enable_tensorboard"]:
            if check_packages.TENSORBOARD_INSTALLED:
                agent_manager_kwargs.update(dict(enable_tensorboard=True))
            else:
                logger.warning(
                    "Option --enable_tensorboard is not available: tensorboard is not installed."
                )

        yield AgentManager(**agent_manager_kwargs)
