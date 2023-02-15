"""
TODO: Test attention modules
"""

import torch
from rlberry.agents.torch.utils.models import MultiLayerPerceptron
from rlberry.agents.torch.utils.models import ConvolutionalNetwork, DuelingNetwork
from rlberry.agents.torch.utils.attention_models import EgoAttention
from rlberry.agents.torch.utils.attention_models import SelfAttention
from rlberry.manager.agent_manager import AgentManager
from rlberry.agents.torch.dqn.dqn import DQNAgent
from rlberry.envs.gym_make import atari_make


def test_mlp():
    model = MultiLayerPerceptron(
        in_size=5, layer_sizes=[10, 10, 10], out_size=10, reshape=False
    )
    x = torch.rand(1, 5)
    y = model.forward(x)
    assert y.shape[1] == 10


def test_mlp_policy():
    model = MultiLayerPerceptron(
        in_size=5, layer_sizes=[10, 10, 10], out_size=10, reshape=False, is_policy=True
    )
    x = torch.rand(1, 5)
    scores = model.action_scores(x)
    assert scores.shape[1] == 10


def test_cnn():
    model = ConvolutionalNetwork(in_channels=10, in_height=20, in_width=30, out_size=15)
    x = torch.rand(1, 10, 20, 30)
    y = model.forward(x)
    assert y.shape[1] == 15


def test_dueling_network():
    model = DuelingNetwork(in_size=10, out_size=15)
    x = torch.rand(1, 10)
    y = model.forward(x)


def test_cnn_policy():
    model = ConvolutionalNetwork(
        in_channels=10, in_height=20, in_width=30, out_size=15, is_policy=True
    )
    x = torch.rand(1, 10, 20, 30)
    scores = model.action_scores(x)
    assert scores.shape[1] == 15


def test_ego_attention():
    module = EgoAttention()
    print(module)


def test_self_attention():
    _ = SelfAttention()


def test_forward_atari_dqn():
    mlp_configs = {
        "type": "MultiLayerPerceptron",  # A network architecture
        "layer_sizes": [512],  # Network dimensions
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

    tuned_agent = AgentManager(
        DQNAgent,  # The Agent class.
        (
            atari_make,
            dict(id="ALE/Breakout-v5", n_envs=3),
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
