import numpy as np
import torch
from gym import spaces
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
    from rlberry.agents.torch.utils.attention_models import EgoAttentionNetwork
    from rlberry.agents.torch.utils.models import MultiLayerPerceptron, DuelingNetwork, ConvolutionalNetwork, \
        Table
    if type == "MultiLayerPerceptron":
        return MultiLayerPerceptron(**kwargs)
    elif type == "DuelingNetwork":
        return DuelingNetwork(**kwargs)
    elif type == "ConvolutionalNetwork":
        return ConvolutionalNetwork(**kwargs)
    elif type == "EgoAttentionNetwork":
        return EgoAttentionNetwork(**kwargs)
    elif type == "Table":
        return Table(**kwargs)
    else:
        raise ValueError("Unknown model type")


def model_factory_from_env(env, **kwargs):
    kwargs = size_model_config(env, **kwargs)
    return model_factory(**kwargs)


def size_model_config(env,
                      **model_config):
    """
    Update the configuration of a model depending on the environment
    observation/action spaces.

    Typically, the input/output sizes.

    Parameters
    ----------
    env : gym.Env
        An environment.
    model_config : dict
        A model configuration.
    """

    if isinstance(env.observation_space, spaces.Box):
        obs_shape = env.observation_space.shape
    elif isinstance(env.observation_space, spaces.Tuple):
        obs_shape = env.observation_space.spaces[0].shape
    elif isinstance(env.observation_space, spaces.Discrete):
        return model_config

    # Assume CHW observation space
    if model_config["type"] == "ConvolutionalNetwork":
        model_config["in_channels"] = int(obs_shape[0])
        model_config["in_height"] = int(obs_shape[1])
        model_config["in_width"] = int(obs_shape[2])
    else:
        model_config["in_size"] = int(np.prod(obs_shape))

    if isinstance(env.action_space, spaces.Discrete):
        model_config["out_size"] = env.action_space.n
    elif isinstance(env.action_space, spaces.Tuple):
        model_config["out_size"] = env.action_space.spaces[0].n
    return model_config


def activation_factory(activation_type):
    if activation_type == "RELU":
        return F.relu
    elif activation_type == "TANH":
        return torch.tanh
    elif activation_type == "ELU":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation_type: {}".format(activation_type))


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
