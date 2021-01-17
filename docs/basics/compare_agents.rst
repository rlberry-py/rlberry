.. _rlberry: https://github.com/rlberry-py/rlberry

Compare different agents
#########################


Two or more agents can be compared using the classes 
:class:`~rlberry.stats.agent_stats.AgentStats` and
:class:`~rlberry.stats.multiple_stats.MultipleStats`, as in the example below.


.. code-block:: python

    from rlberry.envs.classic_control import MountainCar
    from rlberry.agents.reinforce import REINFORCEAgent
    from rlberry.agents.kernel_based.rs_kernel_ucbvi import RSKernelUCBVIAgent
    from rlberry.stats import AgentStats, MultipleStats, plot_episode_rewards


    # Environment
    env = MountainCar()

    # Parameters
    params = {}
    params['reinforce'] = {
            "n_episodes": 300,
            "gamma": 0.99,
            "horizon": 160
    }

    params['kernel'] = {
            "n_episodes": 300,
            "gamma": 0.99,
            "horizon": 160
    }


    # Create AgentStats for REINFORCE and RSKernelUCBVI
    mstats = MultipleStats()
    mstats.append(
        AgentStats(REINFORCEAgent,
                   env,
                   init_kwargs=params['reinforce'],
                   n_fit=4,
                   n_jobs=4)
    )
    mstats.append(
        AgentStats(RSKernelUCBVIAgent,
                   env,
                   init_kwargs=params['kernel'],
                   n_fit=4,
                   n_jobs=4)
    )

    # Fit and plot
    mstats.run()
    plot_episode_rewards(mstats.allstats)

