"""
To run the experiment:

$ python examples/demo_experiment/run.py examples/demo_experiment/params_experiment.yaml

To see more options:

$ python examples/demo_experiment/run.py
"""

from rlberry.experiment import load_experiment_results
from rlberry.experiment import experiment_generator
from rlberry.manager.multiple_managers import MultipleManagers

multimanagers = MultipleManagers()

for agent_manager in experiment_generator():
    multimanagers.append(agent_manager)

    # Alternatively:
    # agent_manager.fit()
    # agent_manager.save()

multimanagers.run()
multimanagers.save()

# Reading the results
del mstats

data = load_experiment_results('results', 'params_experiment')
print(data)
