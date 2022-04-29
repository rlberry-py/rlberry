"""
 =====================
 Demo: demo_from_stable_baselines
 =====================
"""
from rlberry.envs import gym_make
from stable_baselines3 import A2C as A2CStableBaselines
from rlberry.agents import AgentWithSimplePolicy


class A2CAgent(AgentWithSimplePolicy):
    name = "A2C"

    def __init__(
        self,
        env,
        policy,
        learning_rate=7e-4,
        n_steps: int = 200,
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
        **kwargs
    ):
        # init rlberry base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        # rlberry accepts tuples (env_constructor, env_kwargs) as env
        # After a call to __init__, self.env is set as an environment instance
        env = self.env

        # Generate seed for A2CStableBaselines using rlberry seeding
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
            _init_setup_model,
        )

    def fit(self, budget, **kwargs):
        self.wrapped.learn(total_timesteps=budget, **kwargs)

    def policy(self, observation):
        action, _ = self.wrapped.predict(observation, deterministic=True)
        return action

    #
    # For hyperparameter optimization
    #
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
    #
    # Training one agent
    #
    env_ctor = gym_make
    env_kwargs = dict(id="CartPole-v1")
    # env = env_ctor(**env_kwargs)
    # agent = A2CAgent(env, 'MlpPolicy', verbose=1)
    # agent.fit(budget=1000)

    #
    # Training several agents and comparing different hyperparams
    #
    from rlberry.manager import AgentManager, MultipleManagers, evaluate_agents

    stats = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        agent_name="A2C baseline",
        init_kwargs=dict(policy="MlpPolicy", verbose=1),
        fit_kwargs=dict(log_interval=1000),
        fit_budget=2500,
        eval_kwargs=dict(eval_horizon=400),
        n_fit=4,
        parallelization="process",
        output_dir="dev/stable_baselines",
        seed=123,
    )

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
    multimanagers = MultipleManagers(parallelization="thread")
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
