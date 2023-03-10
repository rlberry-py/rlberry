# from rlberry.envs import gym_make
# from rlberry.agents.torch.ppo import PPOAgent


# env = (gym_make, dict(id="Acrobot-v1"))
# # env = gym_make(id="Acrobot-v1")
# ppo = PPOAgent(env)
# ppo.fit(4096)

from rlberry.envs import Wrapper, gym_make
from rlberry.agents.torch import PPOAgent
from rlberry.manager import AgentManager, evaluate_agents
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from gymnasium import make
from rlberry.agents.torch.utils.training import model_factory_from_env
import os
import pathlib
import shutil

from gymnasium.wrappers import StepAPICompatibility

def test_ppo():
    env = "CartPole-v0"
    mdp = make(env)
    mdp = StepAPICompatibility(mdp,output_truncation_bool=False)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=24, n_steps=96),
        n_fit=1,
        agent_name="PPO_rlberry_" + env,
    )

    pporlberry_stats.fit()

    output = evaluate_agents([pporlberry_stats], n_simulations=2, plot=False)
    pporlberry_stats.clear_output_dir()

    env = "Pendulum-v1"
    mdp = make(env)
    mdp = StepAPICompatibility(mdp,output_truncation_bool=False)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=24, n_steps=96),
        n_fit=1,
        agent_name="PPO_rlberry_" + env,
    )

    pporlberry_stats.fit()

    output = evaluate_agents([pporlberry_stats], n_simulations=2, plot=False)
    pporlberry_stats.clear_output_dir()

    env = "Acrobot-v1"
    mdp = make(env)
    mdp = StepAPICompatibility(mdp,output_truncation_bool=False)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=24, n_steps=96),
        n_fit=1,
        agent_name="PPO_rlberry_" + env,
    )

    pporlberry_stats.fit()

    output = evaluate_agents([pporlberry_stats], n_simulations=2, plot=False)
    pporlberry_stats.clear_output_dir()

    # env_ctor = PBall2D
    # env_kwargs = dict()
    env = StepAPICompatibility(PBall2D(),output_truncation_bool=False)
    env_ctor = Wrapper
    env_kwargs = dict(env=env)

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=24, n_steps=96),
        n_fit=1,
        agent_name="PPO_rlberry_" + "PBall2D",
    )

    pporlberry_stats.fit()

    output = evaluate_agents([pporlberry_stats], n_simulations=2, plot=False)
    pporlberry_stats.clear_output_dir()

    # test also non default
    env = "CartPole-v0"
    mdp = make(env)
    mdp = StepAPICompatibility(mdp,output_truncation_bool=False)
    env_ctor = Wrapper
    env_kwargs = dict(env=mdp)

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(
            batch_size=24,
            n_steps=96,
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

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(
            batch_size=24,
            n_steps=96,
            use_gae=False,
            policy_net_fn=model_factory_from_env,
            policy_net_kwargs=dict(
                type="MultiLayerPerceptron",
                layer_sizes=(256,),
                reshape=False,
                is_policy=True,
            ),
            value_net_fn=model_factory_from_env,
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

    # env_ctor = PBall2D
    # env_kwargs = dict()
    env = StepAPICompatibility(PBall2D(),output_truncation_bool=False)
    env_ctor = Wrapper
    env_kwargs = dict(env=env)

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=24, n_steps=96, normalize_rewards=True),
        n_fit=1,
        agent_name="PPO_rlberry_" + "PBall2D",
    )

    pporlberry_stats.fit()

    output = evaluate_agents([pporlberry_stats], n_simulations=2, plot=False)
    pporlberry_stats.clear_output_dir()

    pporlberry_stats = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        fit_budget=int(100),
        eval_kwargs=dict(eval_horizon=2),
        init_kwargs=dict(batch_size=24, n_steps=96, normalize_advantages=True),
        n_fit=1,
        agent_name="PPO_rlberry_" + "PBall2D",
    )

    pporlberry_stats.fit()

    output = evaluate_agents([pporlberry_stats], n_simulations=2, plot=False)
    pporlberry_stats.clear_output_dir()


def test_ppo_classic_env():
    env = gym_make("CartPole-v0")
    agent = PPOAgent(
        env,
        learning_rate=1e-4,
        optimizer_type="ADAM",
    )
    agent.fit(budget=200)

    saving_path = "rlberry/agents/torch/tests/agent_test_ppo_classic_env.pickle"

    # VRemove previous save
    if os.path.exists(saving_path):
        os.remove(saving_path)
    assert not os.path.exists(saving_path)

    # test the save function
    agent.save(saving_path)
    assert os.path.exists(saving_path)

    # test the loading function
    test_load_env = gym_make("CartPole-v0")
    loaded_agent = PPOAgent.load(saving_path, **dict(env=test_load_env))
    assert loaded_agent

    # test the agent
    state,info = test_load_env.reset()
    for tt in range(50):
        action = loaded_agent.policy(state)
        next_state, reward, done, _ = test_load_env.step(action)
        if done:
            next_state,info = test_load_env.reset()
        state = next_state

    os.remove(saving_path)


def test_ppo_agent_manager_classic_env():
    saving_path = "rlberry/agents/torch/tests/agentmanager_test_ppo_classic_env"

    # Remove previous save
    if os.path.exists(saving_path):
        shutil.rmtree(saving_path)
    assert not os.path.exists(saving_path)

    test_agent_manager = AgentManager(
        PPOAgent,  # The Agent class.
        (
            gym_make,
            dict(
                id="CartPole-v0",
            ),
        ),  # The Environment to solve.
        init_kwargs=dict(  # Where to put the agent's hyperparameters
            learning_rate=1e-4,
            optimizer_type="ADAM",
        ),
        fit_budget=200,  # The number of interactions between the agent and the environment during training.
        eval_kwargs=dict(
            eval_horizon=50
        ),  # The number of interactions between the agent and the environment during evaluations.
        n_fit=1,  # The number of agents to train. Usually, it is good to do more than 1 because the training is stochastic.
        agent_name="test_ppo_classic_env",  # The agent's name.
        output_dir=saving_path,
    )

    test_agent_manager.fit(budget=200)

    # test the save function
    test_agent_manager.save()
    assert os.path.exists(saving_path)

    # test the loading function
    test_load_env = gym_make("CartPole-v0")
    path_to_load = next(pathlib.Path(saving_path).glob("**/*.pickle"))
    loaded_agent_manager = AgentManager.load(path_to_load)
    assert loaded_agent_manager

    # test the agent
    state,info = test_load_env.reset()
    for tt in range(50):
        action = loaded_agent_manager.get_agent_instances()[0].policy(state)
        next_s, _, done, test = test_load_env.step(action)
        if done:
            break
        state = next_s

    shutil.rmtree(saving_path)
