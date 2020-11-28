"""
TODO: Test attention modules
"""

import torch
from rlberry.agents.utils.torch_models import MultiLayerPerceptron
from rlberry.agents.utils.torch_models import ConvolutionalNetwork
from rlberry.agents.utils.torch_attention_models import EgoAttention
from rlberry.agents.utils.torch_attention_models import SelfAttention


def test_mlp():
    model = MultiLayerPerceptron(in_size=5,
                                 layer_sizes=[10, 10, 10],
                                 out_size=10,
                                 reshape=False)
    x = torch.rand(1, 5)
    y = model.forward(x)
    assert y.shape[1] == 10


def test_cnn():
    model = ConvolutionalNetwork(in_channels=10,
                                 in_height=20,
                                 in_width=30,
                                 out_size=15)
    x = torch.rand(1, 10, 20, 30)
    y = model.forward(x)
    assert y.shape[1] == 15


def test_ego_attention():
    _ = EgoAttention()


def test_self_attention():
    _ = SelfAttention()
