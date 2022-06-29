# from rlberry.envs import gym_make
# from rlberry.agents.torch.ppo import PPOAgent


# env = (gym_make, dict(id="Acrobot-v1"))
# # env = gym_make(id="Acrobot-v1")
# ppo = PPOAgent(env)
# ppo.fit(4096)

from rlberry.envs import Wrapper
from rlberry.agents.torch import PPOAgent
from rlberry.manager import AgentManager, evaluate_agents
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from gym import make


def test_ppo():

    env = "CartPole-v0"
    mdp = make(env)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=100),
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
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=100),
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
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=100),
        n_fit=1,
        agent_name="PPO_rlberry_" + "PBall2D",
    )

    pporlberry_stats.fit()

    output = evaluate_agents([pporlberry_stats], n_simulations=2, plot=False)
    pporlberry_stats.clear_output_dir()

    # test also non default
    env = "CartPole-v0"
    mdp = make(env)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(
            batch_size=100,
            use_gae=False,
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
        agent_name="PPO_rlberry_" + env,
    )
    pporlberry_stats.fit()

    output = evaluate_agents([pporlberry_stats], n_simulations=2, plot=False)
    pporlberry_stats.clear_output_dir()
