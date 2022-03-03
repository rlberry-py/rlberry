.. _rlberry: https://github.com/rlberry-py/rlberry
.. _`Stable Baselines`: https://github.com/DLR-RM/stable-baselines3

.. _stable_baselines:


Using rlberry_ and `Stable Baselines`_
======================================

`Stable Baselines`_ provides implementations of several Deep RL agents.
rlberry_ provides a wrapper class for `Stable Baselines`_ algorithms, which
makes it easy to train several agents in parallel, optimize hyperparameters,
visualize the results etc.

The wrapper receives metrics from the internal `Stable Baselines`_ and writes
them with the agent's wrapper. As a result, if you're using a custom logger,
set the logger of the wrapped algorithm by calling set_logger.

The example below shows how to use rlberry_ to train several instances of the A2C
implementation of `Stable Baselines`_ and evaluate two hyperparameter configurations.


.. code-block:: python

    from rlberry.envs import gym_make
    from stable_baselines3 import A2C as A2C
    from rlberry.utils.sb_agent import StableBaselinesAgent

    #
    # For hyperparameter optimization
    #
    class A2CAgent(StableBaselinesAgent):
        name = "A2C"

        def __init__(self, env, **kwargs):
            super(A2CAgent, self).__init__(env, algo_cls=A2C, **kwargs)

        @classmethod
        def sample_parameters(cls, trial):
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
            ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
            vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
            normalize_advantage = trial.suggest_categorical(
                "normalize_advantage", [False, True]
            )
            return dict(
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                normalize_advantage=normalize_advantage,
            )


    if __name__ == "__main__":
        env_ctor = gym_make
        env_kwargs = dict(id="CartPole-v1")

        # Training one agent
        env = env_ctor(**env_kwargs)
        agent = StableBaselinesAgent(env, A2C, "MlpPolicy", verbose=1)
        agent.fit(budget=1000)

        # Training several agents and comparing different hyperparams
        from rlberry.manager import AgentManager, MultipleManagers, evaluate_agents

        # Pass the wrapper directly with init_kwargs
        stats = AgentManager(
            StableBaselinesAgent,
            (env_ctor, env_kwargs),
            agent_name="A2C baseline",
            init_kwargs=dict(algo_cls=A2C, policy="MlpPolicy", verbose=1),
            fit_kwargs=dict(log_interval=1000),
            fit_budget=2500,
            eval_kwargs=dict(eval_horizon=400),
            n_fit=4,
            parallelization="process",
            output_dir="dev/stable_baselines",
            seed=123,
        )

        # Pass a subclass for hyperparameter optimization
        stats_alternative = AgentManager(
            A2CAgent,
            (env_ctor, env_kwargs),
            agent_name="A2C optimized",
            init_kwargs=dict(policy="MlpPolicy", verbose=1),
            fit_kwargs=dict(log_interval=1000),
            fit_budget=2500,
            eval_kwargs=dict(eval_horizon=400),
            n_fit=4,
            parallelization="process",
            output_dir="dev/stable_baselines",
            seed=456,
        )

        # Optimize hyperparams (600 seconds)
        stats_alternative.optimize_hyperparams(
            timeout=600,
            n_optuna_workers=2,
            n_fit=2,
            optuna_parallelization="process",
            fit_fraction=1.0,
        )

        # Fit everything in parallel
        multimanagers = MultipleManagers()
        multimanagers.append(stats)
        multimanagers.append(stats_alternative)

        multimanagers.run()

        # Plot policy evaluation
        out = evaluate_agents(multimanagers.managers)
        print(out)

        # Visualize policy
        env = stats_alternative.build_eval_env()
        agent = stats_alternative.agent_handlers[0]
        obs = env.reset()
        for i in range(2500):
            action = agent.policy(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                break
        env.close()
