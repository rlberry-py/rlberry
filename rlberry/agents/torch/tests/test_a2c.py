from rlberry.envs import Wrapper, ResetAPICompatibility
from rlberry.agents.torch import A2CAgent
from rlberry.manager import AgentManager, evaluate_agents
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.envs import gym_make
from gymnasium import make
from gymnasium.wrappers import StepAPICompatibility


def test_a2c():
    env = "CartPole-v0"
    mdp = make(env)
    mdp = StepAPICompatibility(mdp,output_truncation_bool=False)
    mdp = ResetAPICompatibility(mdp)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    a2crlberry_stats = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=100),
        n_fit=1,
        agent_name="A2C_rlberry_" + env,
    )

    a2crlberry_stats.fit()

    output = evaluate_agents([a2crlberry_stats], n_simulations=2, plot=False)
    a2crlberry_stats.clear_output_dir()
    env = "Pendulum-v1"
    mdp = make(env)
    mdp = StepAPICompatibility(mdp,output_truncation_bool=False)
    mdp = ResetAPICompatibility(mdp)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    a2crlberry_stats = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=100),
        n_fit=1,
        agent_name="A2C_rlberry_" + env,
    )

    a2crlberry_stats.fit()

    output = evaluate_agents([a2crlberry_stats], n_simulations=2, plot=False)
    a2crlberry_stats.clear_output_dir()

    env = "Acrobot-v1"
    mdp = make(env)
    mdp = StepAPICompatibility(mdp,output_truncation_bool=False)
    mdp = ResetAPICompatibility(mdp)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    a2crlberry_stats = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=100),
        n_fit=1,
        agent_name="A2C_rlberry_" + env,
    )

    a2crlberry_stats.fit()

    output = evaluate_agents([a2crlberry_stats], n_simulations=2, plot=False)
    a2crlberry_stats.clear_output_dir()
    
    env = StepAPICompatibility(PBall2D(),output_truncation_bool=False)
    env_ctor = Wrapper
    env_kwargs = dict(env=env)

    a2crlberry_stats = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=100),
        n_fit=1,
        agent_name="A2C_rlberry_" + "PBall2D",
    )

    a2crlberry_stats.fit()

    output = evaluate_agents([a2crlberry_stats], n_simulations=2, plot=False)
    a2crlberry_stats.clear_output_dir()

    # test also non default
    env = "CartPole-v0"
    mdp = make(env)
    mdp = StepAPICompatibility(mdp,output_truncation_bool=False)
    mdp = ResetAPICompatibility(mdp)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    a2crlberry_stats = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(
            batch_size=100,
            policy_net_fn="rlberry.agents.torch.utils.training.model_factory_from_env",
            policy_net_kwargs=dict(
                type="MultiLayerPerceptron",
                layer_sizes=(256,),
                reshape=False,
                is_policy=True,
            ),
            value_net_fn="rlberry.agents.torch.utils.training.model_factory_from_env",
            value_net_kwargs=dict(
                type="MultiLayerPerceptron",
                layer_sizes=[
                    512,
                ],
                reshape=False,
                out_size=1,
            ),
        ),
        n_fit=1,
        agent_name="A2C_rlberry_" + env,
    )
    a2crlberry_stats.fit()

    output = evaluate_agents([a2crlberry_stats], n_simulations=2, plot=False)
    a2crlberry_stats.clear_output_dir()
