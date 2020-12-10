import torch
from torch import nn as nn
from torch.nn import functional as F


def loss_function_factory(loss_function):
    if loss_function == "l2":
        return F.mse_loss
    elif loss_function == "l1":
        return F.l1_loss
    elif loss_function == "smooth_l1":
        return F.smooth_l1_loss
    elif loss_function == "bce":
        return F.binary_cross_entropy
    else:
        raise ValueError("Unknown loss function : {}".format(loss_function))


def optimizer_factory(params, optimizer_type="ADAM", **kwargs):
    if optimizer_type == "ADAM":
        return torch.optim.Adam(params=params, **kwargs)
    elif optimizer_type == "RMS_PROP":
        return torch.optim.RMSprop(params=params, **kwargs)
    else:
        raise ValueError("Unknown optimizer type: {}".format(optimizer_type))


def model_factory(type="MultiLayerPerceptron", **kwargs) -> nn.Module:
    from rlberry.agents.utils.torch_attention_models import EgoAttentionNetwork
    from rlberry.agents.utils.torch_models import MultiLayerPerceptron, DuelingNetwork, ConvolutionalNetwork, \
        PolicyConvolutionalNetwork
    if type == "MultiLayerPerceptron":
        return MultiLayerPerceptron(**kwargs)
    elif type == "DuelingNetwork":
        return DuelingNetwork(**kwargs)
    elif type == "ConvolutionalNetwork":
        return ConvolutionalNetwork(**kwargs)
    elif type == "PolicyConvolutionalNetwork":
        return PolicyConvolutionalNetwork(**kwargs)
    elif type == "EgoAttentionNetwork":
        return EgoAttentionNetwork(**kwargs)
    else:
        raise ValueError("Unknown model type")
