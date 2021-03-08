.. _rlberry: https://github.com/rlberry-py/rlberry

.. _evaluate_agent:


Evaluate an agent and optimize its hyperparameters
==================================================

With rlberry_, once you created your agent, it is very easy to train in parallel
several instances of it, analyze the results and optimize hyperparameters. 

This is one of the purposes of the :class:`~rlberry.stats.agent_stats.AgentStats` class,
as shown in the examples below.

.. code-block:: python

    from rlberry.envs import gym_make
    from rlberry.agents.torch.reinforce import REINFORCEAgent
    from rlberry.stats import AgentStats, plot_episode_rewards


    # Environment
    env = gym_make('CartPole-v1')

    # Initial set of parameters
    params = {"n_episodes": 400,
              "gamma": 0.99,
              "horizon": 500}

    # Create AgentStats to fit 4 instances of REINFORCE using 4 threads
    stats = AgentStats(REINFORCEAgent,
                       env,
                       init_kwargs=params,
                       n_fit=4,
                       n_jobs=4)

    # Fit the 4 instances
    stats.fit()

    # The fit() method of REINFORCEAgent returns
    # a dictionary `info` such that info['episode_rewards']
    # is a numpy array with the sum of rewards obtained in
    # each episode.
    # The method below can be used to plot it!
    plot_episode_rewards(stats)


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
        batch_size = trial.suggest_categorical('batch_size', [1, 4, 8, 16, 32])
        gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.99])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
        entr_coef = trial.suggest_loguniform('entr_coef', 1e-8, 0.1)

        return {
                'batch_size': batch_size,
                'gamma': gamma,
                'learning_rate': learning_rate,
                'entr_coef': entr_coef,
                }


Now we can use the :meth:`optimize_hyperparams` method 
of :class:`~rlberry.stats.agent_stats.AgentStats` to find good parameters for our agent:

.. code-block:: python

    # Run optimization and print results
    stats.optimize_hyperparams(
        n_trials=100,
        timeout=10,   # stop after 10 seconds
        n_sim=5,
        n_fit=2,
        n_jobs=2,
        sampler_method='optuna_default'
        )

    print(stats.best_hyperparams)

    # Calling fit() again will train the agent with the optimized parameters
    stats.fit()
    plot_episode_rewards(stats)
