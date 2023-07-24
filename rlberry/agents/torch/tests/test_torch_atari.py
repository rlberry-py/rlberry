from rlberry.manager.agent_manager import ExperimentManager
from rlberry.agents.torch.dqn.dqn import DQNAgent
from rlberry.envs.gym_make import atari_make

from rlberry.manager.agent_manager import ExperimentManager
from rlberry.agents.torch import PPOAgent
from rlberry.agents.torch.utils.training import model_factory_from_env
import pathlib
import numpy as np
import pytest
import os

import tempfile


def test_forward_dqn():
    mlp_configs = {
        "type": "MultiLayerPerceptron",  # A network architecture
        "layer_sizes": [32],  # Network dimensions
        "reshape": False,
        "is_policy": False,  # The network should output a distribution
        # over actions
    }

    cnn_configs = {
        "type": "ConvolutionalNetwork",  # A network architecture
        "activation": "RELU",
        "in_channels": 4,
        "in_height": 84,
        "in_width": 84,
        "head_mlp_kwargs": mlp_configs,
        "transpose_obs": False,
        "is_policy": False,  # The network should output a distribution
    }

    tuned_agent = ExperimentManager(
        DQNAgent,  # The Agent class.
        (
            atari_make,
            # uncomment when rlberry will manage vectorized env
            # dict(id="ALE/Breakout-v5", n_envs=3),
            dict(id="ALE/Breakout-v5", n_envs=1),
        ),  # The Environment to solve.
        init_kwargs=dict(  # Where to put the agent's hyperparameters
            q_net_constructor="rlberry.agents.torch.utils.training.model_factory_from_env",
            q_net_kwargs=cnn_configs,
            max_replay_size=100,
            batch_size=32,
            learning_starts=100,
            gradient_steps=1,
            epsilon_final=0.01,
            learning_rate=1e-4,  # Size of the policy gradient descent steps.
            chunk_size=5,
        ),
        fit_budget=200,  # The number of interactions between the agent and the environment during training.
        eval_kwargs=dict(
            eval_horizon=10
        ),  # The number of interactions between the agent and the environment during evaluations.
        n_fit=1,  # The number of agents to train. Usually, it is good to do more than 1 because the training is stochastic.
        agent_name="DQN_test",  # The agent's name.
    )

    tuned_agent.fit()


def test_forward_empty_input_dim():
    mlp_configs = {
        "type": "MultiLayerPerceptron",  # A network architecture
        "layer_sizes": [32],  # Network dimensions
        "reshape": False,
        "is_policy": False,  # The network should output a distribution
        # over actions
    }

    cnn_configs = {
        "type": "ConvolutionalNetwork",  # A network architecture
        "activation": "RELU",
        "head_mlp_kwargs": mlp_configs,
        "transpose_obs": False,
        "is_policy": False,  # The network should output a distribution
    }

    tuned_agent = ExperimentManager(
        DQNAgent,  # The Agent class.
        (
            atari_make,
            # uncomment when rlberry will manage vectorized env
            # dict(id="ALE/Breakout-v5", n_envs=3),
            dict(id="ALE/Breakout-v5", n_envs=1),
        ),  # The Environment to solve.
        init_kwargs=dict(  # Where to put the agent's hyperparameters
            q_net_constructor="rlberry.agents.torch.utils.training.model_factory_from_env",
            q_net_kwargs=cnn_configs,
            max_replay_size=100,
            batch_size=32,
            learning_starts=100,
            gradient_steps=1,
            epsilon_final=0.01,
            learning_rate=1e-4,  # Size of the policy gradient descent steps.
            chunk_size=5,
        ),
        fit_budget=10,  # The number of interactions between the agent and the environment during training.
        eval_kwargs=dict(
            eval_horizon=10
        ),  # The number of interactions between the agent and the environment during evaluations.
        n_fit=1,  # The number of agents to train. Usually, it is good to do more than 1 because the training is stochastic.
        agent_name="DQN_test",  # The agent's name.
    )

    tuned_agent.fit()


@pytest.mark.parametrize("num_envs", [1, 3])
def test_ppo_vectorized_atari_env(num_envs):
    policy_mlp_configs = {
        "type": "MultiLayerPerceptron",  # A network architecture
        "layer_sizes": [32],  # Network dimensions
        "reshape": False,
        "is_policy": True,  # The network should output a distribution
        # over actions
    }

    critic_mlp_configs = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": [32],
        "reshape": False,
        "out_size": 1,  # The critic network is an approximator of
        # a value function V: States -> |R
    }

    policy_configs = {
        "type": "ConvolutionalNetwork",  # A network architecture
        "activation": "RELU",
        "in_channels": 4,
        "in_height": 84,
        "in_width": 84,
        "head_mlp_kwargs": policy_mlp_configs,
        "transpose_obs": False,
        "is_policy": True,  # The network should output a distribution
    }

    critic_configs = {
        "type": "ConvolutionalNetwork",
        "layer_sizes": "RELU",
        "in_channels": 4,
        "in_height": 84,
        "in_width": 84,
        "head_mlp_kwargs": critic_mlp_configs,
        "transpose_obs": False,
        "out_size": 1,
    }

    agent = PPOAgent(
        (
            atari_make,
            dict(id="ALE/Freeway-v5"),
        ),
        optimizer_type="ADAM",  # What optimizer to use for policy gradient descent steps.
        learning_rate=1e-4,  # Size of the policy gradient descent steps.
        policy_net_fn=model_factory_from_env,  # A policy network constructor
        policy_net_kwargs=policy_configs,  # Policy network's architecure
        value_net_fn=model_factory_from_env,  # A Critic network constructor
        value_net_kwargs=critic_configs,  # Critic network's architecure.
        n_envs=num_envs,
        n_steps=64,
        batch_size=128,
        # **dict(eval_env=(atari_make,dict(id="ALE/Freeway-v5",n_envs=1)))
    )
    agent.fit(budget=500)

    with tempfile.TemporaryDirectory() as tmpdirname:
        saving_path = tmpdirname + "/agent_test_ppo_vect_env.pickle"

        # test the save function
        agent.save(saving_path)
        assert os.path.exists(saving_path)

        # test the loading function
        test_load_env = atari_make("ALE/Freeway-v5")
        test_load_env.reset()
        loaded_agent = PPOAgent.load(
            saving_path, **dict(env=test_load_env), copy_env=False
        )
        assert loaded_agent

        # test the agent
        observation, info = test_load_env.reset()
        for tt in range(100):
            action = loaded_agent.policy(observation)
            next_observation, reward, terminated, truncated, info = test_load_env.step(
                action
            )
            done = terminated or truncated
            if done:
                next_observation, info = test_load_env.reset()
            observation = next_observation


@pytest.mark.parametrize("num_envs", [1, 3])
def test_ppo_agent_manager_vectorized_atari_env(num_envs):
    with tempfile.TemporaryDirectory() as tmpdirname:
        saving_path = tmpdirname + "/agentmanager_test_ppo_vectorized_env"

        policy_mlp_configs = {
            "type": "MultiLayerPerceptron",  # A network architecture
            "layer_sizes": [32],  # Network dimensions
            "reshape": False,
            "is_policy": True,  # The network should output a distribution
            # over actions
        }

        critic_mlp_configs = {
            "type": "MultiLayerPerceptron",
            "layer_sizes": [32],
            "reshape": False,
            "out_size": 1,  # The critic network is an approximator of
            # a value function V: States -> |R
        }

        policy_configs = {
            "type": "ConvolutionalNetwork",  # A network architecture
            "activation": "RELU",
            "in_channels": 4,
            "in_height": 84,
            "in_width": 84,
            "head_mlp_kwargs": policy_mlp_configs,
            "transpose_obs": False,
            "is_policy": True,  # The network should output a distribution
        }

        critic_configs = {
            "type": "ConvolutionalNetwork",
            "layer_sizes": "RELU",
            "in_channels": 4,
            "in_height": 84,
            "in_width": 84,
            "head_mlp_kwargs": critic_mlp_configs,
            "transpose_obs": False,
            "out_size": 1,
        }

        test_agent_manager = ExperimentManager(
            PPOAgent,  # The Agent class.
            (
                atari_make,
                dict(id="ALE/Atlantis-v5"),
            ),  # The Environment to solve.
            init_kwargs=dict(  # Where to put the agent's hyperparameters
                optimizer_type="ADAM",  # What optimizer to use for policy gradient descent steps.
                learning_rate=1e-4,  # Size of the policy gradient descent steps.
                policy_net_fn=model_factory_from_env,  # A policy network constructor
                policy_net_kwargs=policy_configs,  # Policy network's architecure
                value_net_fn=model_factory_from_env,  # A Critic network constructor
                value_net_kwargs=critic_configs,  # Critic network's architecure.
                n_envs=num_envs,
                n_steps=64,
                batch_size=128,
            ),
            fit_budget=200,  # The number of interactions between the agent and the environment during training.
            eval_kwargs=dict(
                eval_horizon=50
            ),  # The number of interactions between the agent and the environment during evaluations.
            n_fit=1,  # The number of agents to train. Usually, it is good to do more than 1 because the training is stochastic.
            agent_name="test_ppo_vectorized_env",  # The agent's name.
            output_dir=saving_path,
            # eval_env = (atari_make,dict(id="ALE/Atlantis-v5",n_envs=1))
        )
        test_agent_manager.fit(budget=500)

        # test the save function
        test_agent_manager.save()
        assert os.path.exists(saving_path)

        # test the loading function
        test_load_env = atari_make("ALE/Atlantis-v5")
        test_load_env.reset()
        path_to_load = next(pathlib.Path(saving_path).glob("**/*.pickle"))
        loaded_agent_manager = ExperimentManager.load(path_to_load)
        assert loaded_agent_manager

        # test the agent
        obs, infos = test_load_env.reset()
        for tt in range(50):
            actions = loaded_agent_manager.get_agent_instances()[0].policy(obs)
            obs, reward, terminated, truncated, info = test_load_env.step(actions)
            done = np.logical_or(terminated, truncated)
            if done:
                break
