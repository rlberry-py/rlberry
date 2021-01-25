"""
To run the experiment:

$ python examples/demo_experiment/run.py examples/demo_experiment/params_experiment.yaml

To see more options:

$ python examples/demo_experiment/run.py
"""

from rlberry.experiment import experiment_generator
from rlberry.stats.multiple_stats import MultipleStats

mstats = MultipleStats()

for agent_stats in experiment_generator():
    mstats.append(agent_stats)

    # Alternatively:
    # agent_stats.fit()
    # agent_stats.save_results()
    # agent_stats.save()

mstats.run()
mstats.save()


# Reading the results
del mstats
from rlberry.experiment import load_experiment_results

data = load_experiment_results('results', 'params_experiment')
print(data)
