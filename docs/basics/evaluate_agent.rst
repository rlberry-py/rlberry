.. _rlberry: https://github.com/rlberry-py/rlberry

.. _evaluate_agent:


Evaluate an agent and optimize its hyperparameters
==================================================

With rlberry_, once you created your agent, it is very easy to train in parallel
several instances of it, analyze the results and optimize hyperparameters.

This is one of the purposes of the :class:`~rlberry.manager.agent_manager.AgentManager` class,
as shown in the examples below.

.. code-block:: python

    from rlberry.envs import gym_make
    from rlberry.agents.torch.reinforce import REINFORCEAgent
    from rlberry.manager import AgentManager, plot_writer_data


    # Environment (constructor, kwargs)
    env = (gym_make, dict(id="CartPole-v1"))

    # Initial set of parameters
    params = dict(
        gamma=0.99,
        horizon=500,
    )

    fit_budget = 200  # number of episodes to fit the agent

    eval_kwargs = dict(eval_horizon=500)  # parameters to evaluate the agent


    # Create AgentManager to fit 4 instances of REINFORCE in parallel.
    stats = AgentManager(
        REINFORCEAgent,
        env,
        init_kwargs=params,
        eval_kwargs=eval_kwargs,
        fit_budget=fit_budget,
        n_fit=4,
        parallelization="thread",
    )

    # Fit the 4 instances
    stats.fit()

    # The fit() method of REINFORCEAgent logs data to a :class:`~rlberry.utils.writers.DefaultWriter`
    # object. The method below can be used to plot those data!
    plot_writer_data(stats, tag="episode_rewards")



To run hyperparameter optimization, the agent class needs to implement a
:meth:`sample_paratemers` method (see :class:`~rlberry.agents.agent.Agent` class).

For :class:`~rlberry.agents.reinforce.reinforce.REINFORCEAgent`, this method looks like:

.. code-block:: python

    @classmethod
    def sample_parameters(cls, trial):
        """
        Sample hyperparameters for hyperparam optimization using
        Optuna (https://optuna.org/)

        Note: only the kwargs sent to __init__ are optimized. Make sure to
        include in the Agent constructor all "optimizable" parameters.

        Parameters
        ----------
        trial: optuna.trial
        """
        batch_size = trial.suggest_categorical("batch_size", [1, 4, 8, 16, 32])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        entr_coef = trial.suggest_float("entr_coef", 1e-8, 0.1, log=True)

        return {
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "entr_coef": entr_coef,
        }


Now we can use the :meth:`optimize_hyperparams` method
of :class:`~rlberry.manager.agent_manager.AgentManager` to find good parameters for our agent:

.. code-block:: python

    # Run optimization and print results
    stats.optimize_hyperparams(
        n_trials=100,
        timeout=10,  # stop after 10 seconds
        n_fit=2,
        sampler_method="optuna_default",
    )

    print(stats.best_hyperparams)

    # Calling fit() again will train the agent with the optimized parameters
    stats.fit()
    plot_writer_data(stats, tag="episode_rewards")
