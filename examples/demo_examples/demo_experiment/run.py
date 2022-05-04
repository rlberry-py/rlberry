"""
 =====================
 Demo: run
 =====================
"""
"""
To run the experiment:

$ python examples/demo_examples/demo_experiment/run.py examples/demo_examples/demo_experiment/params_experiment.yaml

To see more options:

$ python examples/demo_examples/demo_experiment/run.py
"""

from rlberry.experiment import load_experiment_results
from rlberry.experiment import experiment_generator
from rlberry.manager.multiple_managers import MultipleManagers


if __name__ == "__main__":
    multimanagers = MultipleManagers(parallelization="thread")

    for agent_manager in experiment_generator():
        multimanagers.append(agent_manager)

    multimanagers.run()
    multimanagers.save()

    # Reading the results
    del multimanagers

    data = load_experiment_results("results", "params_experiment")

    print(data)

    # Fit one of the managers for a few more episodes
    # If tensorboard is enabled, you should see more episodes ran for 'rsucbvi_alternative'
    data["manager"]["rsucbvi_alternative"].fit(50)
