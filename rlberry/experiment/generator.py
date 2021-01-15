"""Run experiments.

Usage:
  run.py <experiment_path> [--writer] [--n_fit=<nf>] [--n_jobs=<nj>] [--output_dir=<dir>]
  run.py (-h | --help)

Options:
  -h --help     Show this screen.
  --writer      Use a tensorboard writer.
  --n_fit=<nf>  Number of times each agent is fit [default: 4].
  --n_jobs=<nj>  Number of jobs used to fit each agent [default: 4].
  --output_dir=<dir>  Directory to save the results [default: results].
"""
from docopt import docopt
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from rlberry.experiment.yaml_utils import parse_experiment_config
from rlberry.seeding.seeding import set_global_seed


doc = __doc__


def experiment_generator():
    """
    Parse command line arguments, set global seed and yields
    AgentStats instances.
    """
    args = docopt(__doc__)
    for (seed, agent_stats) in parse_experiment_config(
                Path(args["<experiment_path>"]),
                n_fit=int(args["--n_fit"]),
                n_jobs=int(args["--n_jobs"]),
                output_base_dir=args["--output_dir"]):
        set_global_seed(seed)
        if args["--writer"]:
            writer_fn = lambda logdir: SummaryWriter(logdir)
            for idx in range(agent_stats.n_fit):
                logdir = agent_stats.output_dir / f"run_{idx + 1}_{datetime.now().strftime('%b%d_%H-%M-%S')}"
                agent_stats.set_writer(idx=idx, writer_fn=writer_fn, writer_kwargs=dict(logdir=logdir))

        yield agent_stats
