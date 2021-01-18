.. _rlberry: https://github.com/rlberry-py/rlberry
.. _`Stable Baselines`: https://github.com/DLR-RM/stable-baselines3


Using rlberry_ and `Stable Baselines`_
#######################################

`Stable Baselines`_ provides implementations of several Deep RL agents.
It is very easy to train their agents with rlberry_, which we can use to
train several agents in parallel, optimize hyperparameters, visualize the results etc.

The example below shows how to use rlberry_ to train several instances of the A2C
implementation of `Stable Baselines`_ and evaluate two hyperparameter configurations.


.. code-block:: python

    from rlberry.envs import gym_make
    from stable_baselines3 import A2C as A2CStableBaselines
    from rlberry.agents import Agent

    import rlberry.seeding as seeding


    class A2CAgent(Agent):
        """
        Wraps stable_baselines3's A2C into an rlberry Agent.
        """

        name = 'A2C'

        def __init__(self,
                    env,
                    policy,
                    learning_rate=7e-4,
                    n_steps: int = 5,
                    gamma: float = 0.99,
                    gae_lambda: float = 1.0,
                    ent_coef: float = 0.0,
                    vf_coef: float = 0.5,
                    max_grad_norm: float = 0.5,
                    rms_prop_eps: float = 1e-5,
                    use_rms_prop: bool = True,
                    use_sde: bool = False,
                    sde_sample_freq: int = -1,
                    normalize_advantage: bool = False,
                    tensorboard_log=None,
                    create_eval_env=False,
                    policy_kwargs=None,
                    verbose: int = 0,
                    seed=None,
                    device="auto",
                    _init_setup_model: bool = True,
                    **kwargs):

            # Generate seed for A2CStableBaselines using rlberry seeding
            self.rng = seeding.get_rng()
            seed = self.rng.integers(2**32).item()

            # init stable baselines class
            self.wrapped = A2CStableBaselines(
                policy,
                env,
                learning_rate,
                n_steps,
                gamma,
                gae_lambda,
                ent_coef,
                vf_coef,
                max_grad_norm,
                rms_prop_eps,
                use_rms_prop,
                use_sde,
                sde_sample_freq,
                normalize_advantage,
                tensorboard_log,
                create_eval_env,
                policy_kwargs,
                verbose,
                seed,
                device,
                _init_setup_model)

            # init rlberry base class
            Agent.__init__(self, env, **kwargs)

        def fit(self, **kwargs):
            result = self.wrapped.learn(**kwargs)
            info = {}  # possibly store something from results
            return info

        def policy(self, observation, **kwargs):
            action, _state = self.wrapped.predict(observation, **kwargs)
            return action


    #
    # Traning one agent
    #


    env = gym_make('CartPole-v1')
    agent = A2CAgent(env, 'MlpPolicy', verbose=1)
    agent.fit(total_timesteps=1000)

    obs = env.reset()
    for i in range(1000):
        action = agent.policy(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
    env.close()

    #
    # Traning several agents and comparing different hyperparams
    #
    from rlberry.stats import AgentStats, MultipleStats, agent_stats, compare_policies

    stats = AgentStats(
        A2CAgent,
        env,
        eval_horizon=200,
        agent_name='A2C baseline',
        init_kwargs={'policy': 'MlpPolicy', 'verbose': 1},
        fit_kwargs={'total_timesteps': 1000},
        policy_kwargs={'deterministic': True},
        n_fit=4,
        n_jobs=4,
        joblib_backend='loky')   # we might need 'threading' here, since stable baselines creates processes
                                # 'multiprocessing' does not work, 'loky' seems good

    stats_alternative = AgentStats(
        A2CAgent,
        env,
        eval_horizon=200,
        agent_name='A2C high learning rate',
        init_kwargs={'policy': 'MlpPolicy', 'verbose': 1, 'learning_rate': 0.01},
        fit_kwargs={'total_timesteps': 1000},
        policy_kwargs={'deterministic': True},
        n_fit=4,
        n_jobs=4,
        joblib_backend='loky')

    # Fit everything in parallel
    mstats = MultipleStats()
    mstats.append(stats)
    mstats.append(stats_alternative)

    mstats.run()

    # Plot policy evaluation
    compare_policies(mstats.allstats)


.. warning::
    When using :class:`~rlberry.stats.agent_stats.AgentStats` with
    `Stable Baselines`_ agents, make sure to set :code:`joblib_backend='loky'`
    or  :code:`joblib_backend='threading'` to avoid conflicts with the
    multiprocessing used by `VecEnv <https://github.com/DLR-RM/stable-baselines3/blob/18d10dbf42dd6dff6d457b45b521fdf2a1169a7e/stable_baselines3/common/vec_env/subproc_vec_env.py>`_ 
    in Stable Baselines.

