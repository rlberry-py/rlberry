"""
 =====================
 Demo: demo_stable_baselines
 =====================
"""
from rlberry.envs import gym_make
from stable_baselines3 import A2C as A2C
from rlberry.agents import StableBaselinesAgent


# Class for hyperparameter optimization
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
    observation,info = env.reset()
    for i in range(2500):
        action = agent.policy(observation)
        observation, reward, terminated, truncated, info  = env.step(action)
        done = terminated or truncated
        env.render()
        if done:
            break
    env.close()
