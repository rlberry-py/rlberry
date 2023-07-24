"""
TODO: Test attention modules
"""

import torch

from rlberry.agents.torch.utils.models import (
    ConvolutionalNetwork,
    DuelingNetwork,
    MultiLayerPerceptron,
)


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
