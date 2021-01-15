"""
To run the experiment:

$ python examples/demo_experiment/run.py examples/demo_experiment/params_experiment.yaml

To see more options:

$ python examples/demo_experiment/run.py
"""

from pathlib import Path
from rlberry.experiment import experiment_generator

for agent_stats in experiment_generator():
    print(agent_stats)
    agent_stats.fit()
    agent_stats.save_results()
    agent_stats.save(Path(agent_stats.output_dir) / 'stats')
