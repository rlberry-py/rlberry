.. _rlberry: https://github.com/rlberry-py/rlberry

Setup and run experiments using yaml config files
##################################################

**experiment.yaml**

.. code-block:: yaml

    description: 'RSUCBVI in NRoom Environment'
    seed: 123
    n_episodes: 100
    horizon: 50
    train_env: 'room.yaml'
    eval_env: 'room.yaml'
    agents:
        - 'rsucbvi.yaml'
        - 'rsucbvi_alternative.yaml'


**room.yaml**

.. code-block:: yaml

    constructor: 'rlberry.envs.benchmarks.grid_exploration.nroom.NRoom'
    params:
        array_observation: true
        nrooms: 5

**rsucbvi.yaml**

.. code-block:: yaml

    agent_class: 'rlberry.agents.kernel_based.rs_ucbvi.RSUCBVIAgent'
    gamma: 1.0
    lp_metric: 2
    min_dist: 0.0
    max_repr: 800
    bonus_scale_factor: 1.0
    reward_free: True


**rsucbvi_alternative.yaml**

.. code-block:: yaml

    base_config: 'rsucbvi.yaml'
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
