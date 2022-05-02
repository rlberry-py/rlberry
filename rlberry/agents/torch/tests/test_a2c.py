from rlberry.envs import Wrapper
from rlberry.agents.torch import A2CAgent
from rlberry.manager import AgentManager, evaluate_agents
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from gym import make


def test_a2c():

    env = "CartPole-v0"
    mdp = make(env)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    a2crlberry_stats = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(2),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(horizon=2),
        n_fit=1,
        agent_name="A2C_rlberry_" + env,
    )

    a2crlberry_stats.fit()

    output = evaluate_agents([a2crlberry_stats], n_simulations=2, plot=False)
    a2crlberry_stats.clear_output_dir()

    env = "Acrobot-v1"
    mdp = make(env)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    a2crlberry_stats = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(2),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(horizon=2),
        n_fit=1,
        agent_name="A2C_rlberry_" + env,
    )

    a2crlberry_stats.fit()

    output = evaluate_agents([a2crlberry_stats], n_simulations=2, plot=False)
    a2crlberry_stats.clear_output_dir()

    env_ctor = PBall2D
    env_kwargs = dict()

    a2crlberry_stats = AgentManager(
        A2CAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(2),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(horizon=2),
        n_fit=1,
        agent_name="A2C_rlberry_" + "PBall2D",
    )

    a2crlberry_stats.fit()

    output = evaluate_agents([a2crlberry_stats], n_simulations=2, plot=False)
    a2crlberry_stats.clear_output_dir()
