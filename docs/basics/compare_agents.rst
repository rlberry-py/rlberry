.. _rlberry: https://github.com/rlberry-py/rlberry

.. _compare_agents:


Compare different agents
========================


Two or more agents can be compared using the classes 
:class:`~rlberry.stats.agent_stats.AgentStats` and
:class:`~rlberry.stats.multiple_stats.MultipleStats`, as in the example below.


.. code-block:: python

        import numpy as np
        from rlberry.envs.classic_control import MountainCar
        from rlberry.agents.torch.reinforce import REINFORCEAgent
        from rlberry.agents.kernel_based.rs_kernel_ucbvi import RSKernelUCBVIAgent
        from rlberry.stats import AgentStats, MultipleStats, plot_writer_data


        # Environment constructor and kwargs
        env = (MountainCar, {})

        # Parameters
        params = {}
        params['reinforce'] = dict(
        gamma=0.99,
        horizon=160,
        )

        params['kernel'] = dict(
        gamma=0.99,
        horizon=160,
        )

        eval_kwargs = dict(eval_horizon=200)

        # Create AgentStats for REINFORCE and RSKernelUCBVI
        mstats = MultipleStats()
        mstats.append(
        AgentStats(
                REINFORCEAgent,
                env,
                init_kwargs=params['reinforce'],
                fit_budget=100,
                n_fit=4,
                parallelization='thread')
        )
        mstats.append(
        AgentStats(
                RSKernelUCBVIAgent,
                env,
                init_kwargs=params['kernel'],
                fit_budget=100,
                n_fit=4,
                parallelization='thread')
        )

        # Fit and plot
        mstats.run()
        plot_writer_data(
        mstats.allstats,
        tag='episode_rewards',
        preprocess_func=np.cumsum,
        title="Cumulative Rewards")

