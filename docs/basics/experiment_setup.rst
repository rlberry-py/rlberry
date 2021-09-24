.. _rlberry: https://github.com/rlberry-py/rlberry

.. _experiment_setup:


Setup and run experiments using yaml config files
=================================================


To setup an experiment with rlberry, you can use yaml files. You'll need:

* An **experiment.yaml** with some global parameters: seed, number of episodes, horizon, environments (for training and evaluation) and a list of agents to run.

* yaml files describing the environments and the agents

* A main python script that reads the files and generates :class:`~rlberry.manager.agent_manager.AgentManager` instances to run each agent.


This can be done very succinctly as in the example below:


**experiment.yaml**

.. code-block:: yaml

    description: 'RSUCBVI in NRoom'
    seed: 123
    train_env: 'examples/demo_experiment/room.yaml'
    eval_env: 'examples/demo_experiment/room.yaml'
    agents:
        - 'examples/demo_experiment/rsucbvi.yaml'
        - 'examples/demo_experiment/rsucbvi_alternative.yaml'


**room.yaml**

.. code-block:: yaml

    constructor: 'rlberry.envs.benchmarks.grid_exploration.nroom.NRoom'
    params:
        reward_free: false
        array_observation: true
        nrooms: 5

**rsucbvi.yaml**

.. code-block:: yaml

    agent_class: 'rlberry.agents.kernel_based.rs_ucbvi.RSUCBVIAgent'
    init_kwargs:
        gamma: 1.0
        lp_metric: 2
        min_dist: 0.0
        max_repr: 800
        bonus_scale_factor: 1.0
        reward_free: True
        horizon: 50
    eval_kwargs:
        eval_horizon: 50
    fit_kwargs:
        fit_budget: 100

**rsucbvi_alternative.yaml**

.. code-block:: yaml

    base_config: 'examples/demo_experiment/rsucbvi.yaml'
    init_kwargs:
        gamma: 0.9



**run.py**

.. code-block:: python

    """
    To run the experiment:

    $ python run.py experiment.yaml

    To see more options:

    $ python run.py -h
    """

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
