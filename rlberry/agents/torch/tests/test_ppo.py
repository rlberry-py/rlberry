from rlberry.envs import Wrapper, gym_make
from rlberry.agents.torch import PPOAgent
from rlberry.manager import AgentManager, evaluate_agents
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from gym import make


def test_regression_ppo():
    env = gym_make("CartPole-v0")
    ppo = PPOAgent(**dict(env=env, batch_size=1))
    ppo.fit(4)


def test_ppo():

    env = "CartPole-v0"
    env_ctor = gym_make
    env_kwargs = dict(id=env)

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(20),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(horizon=2, batch_size=1),
        n_fit=1,
        agent_name="PPO_rlberry_" + env,
    )

    pporlberry_stats.fit()

    output = evaluate_agents([pporlberry_stats], n_simulations=2, plot=False)
    pporlberry_stats.clear_output_dir()

    env = "Acrobot-v1"
    mdp = make(env)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(2),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(horizon=2, batch_size=1),
        n_fit=1,
        agent_name="PPO_rlberry_" + env,
    )

    pporlberry_stats.fit()

    output = evaluate_agents([pporlberry_stats], n_simulations=2, plot=False)
    pporlberry_stats.clear_output_dir()

    env_ctor = PBall2D
    env_kwargs = dict()

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(2),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(horizon=2, batch_size=1),
        n_fit=1,
        agent_name="PPO_rlberry_" + "PBall2D",
    )

    pporlberry_stats.fit()

    output = evaluate_agents([pporlberry_stats], n_simulations=2, plot=False)
    pporlberry_stats.clear_output_dir()
