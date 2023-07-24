import numpy as np
import torch
from gymnasium import spaces
from torch import nn as nn
from torch.nn import functional as F


def loss_function_factory(loss_function, **kwargs):
    if loss_function == "l2":
        return torch.nn.MSELoss(**kwargs)
    elif loss_function == "l1":
        return torch.nn.L1Loss(**kwargs)
    elif loss_function == "smooth_l1":
        return torch.nn.SmoothL1Loss(**kwargs)
    elif loss_function == "bce":
        return torch.nn.BCELoss(**kwargs)
    else:
        raise ValueError("Unknown loss function : {}".format(loss_function))


def optimizer_factory(params, optimizer_type="ADAM", **kwargs):
    if optimizer_type == "ADAM":
        return torch.optim.Adam(params=params, **kwargs)
    elif optimizer_type == "RMS_PROP":
        return torch.optim.RMSprop(params=params, **kwargs)
    else:
        raise ValueError("Unknown optimizer type: {}".format(optimizer_type))


def model_factory_from_env(env, **kwargs):
    """Returns a torch module after setting up input/output dimensions according to an env.

    Parameters
    ----------
    env: gym.Env
        Environment
    **kwargs: Dict
        Parameters to be updated, used to call :func:`~rlberry.agents.torch.utils.training.model_factory`.
    """
    kwargs = size_model_config(env, **kwargs)
    return model_factory(**kwargs)


def model_factory(type="MultiLayerPerceptron", **kwargs) -> nn.Module:
    """Build a neural net of a given type.

    Parameters
    ----------
    type: {"MultiLayerPerceptron",
           "ConvolutionalNetwork",
           "DuelingNetwork",
           "Table"}, default = "MultiLayerPerceptron"
        Type of neural network.
    **kwargs: dict
        Parameters that vary according to each neural net type, see

        * :class:`~rlberry.agents.torch.utils.models.MultiLayerPerceptron`

        * :class:`~rlberry.agents.torch.utils.models.ConvolutionalNetwork`

        * :class:`~rlberry.agents.torch.utils.models.DuelingNetwork`

        * :class:`~rlberry.agents.torch.utils.models.Table`
    """
    from rlberry.agents.torch.utils.models import (
        ConvolutionalNetwork,
        DuelingNetwork,
        MultiLayerPerceptron,
        Table,
    )

    if type == "MultiLayerPerceptron":
        return MultiLayerPerceptron(**kwargs)
    elif type == "DuelingNetwork":
        return DuelingNetwork(**kwargs)
    elif type == "ConvolutionalNetwork":
        return ConvolutionalNetwork(**kwargs)
    elif type == "Table":
        return Table(**kwargs)
    else:
        raise ValueError("Unknown model type")


def size_model_config(env, **model_config):
    """
    Setup input/output dimensions for the configuration of
    a model depending on the environment observation/action spaces.

    Parameters
    ----------
    env : gym.Env
        An environment.
    model_config : dict
        Parameters to be updated, used to call :func:`~rlberry.agents.torch.utils.training.model_factory`.
        If "out_size" is not given in model_config, assumes
        that the output dimension of the neural net is equal to the number
        of actions in the environment.
    """

    if isinstance(env.observation_space, spaces.Box):
        obs_shape = env.observation_space.shape
    elif isinstance(env.observation_space, spaces.Tuple):
        obs_shape = env.observation_space.spaces[0].shape
    elif isinstance(env.observation_space, spaces.Discrete):
        return model_config

    # Assume CHW observation space
    if "type" in model_config and model_config["type"] == "ConvolutionalNetwork":
        if "transpose_obs" in model_config and not model_config["transpose_obs"]:
            # Assume CHW observation space
            if "in_channels" not in model_config:
                model_config["in_channels"] = int(obs_shape[0])
            if "in_height" not in model_config:
                model_config["in_height"] = int(obs_shape[1])
            if "in_width" not in model_config:
                model_config["in_width"] = int(obs_shape[2])
        else:
            # Assume WHC observation space to transpose
            if "in_channels" not in model_config:
                model_config["in_channels"] = int(obs_shape[2])
            if "in_height" not in model_config:
                model_config["in_height"] = int(obs_shape[1])
            if "in_width" not in model_config:
                model_config["in_width"] = int(obs_shape[0])
    else:
        model_config["in_size"] = int(np.prod(obs_shape))

    if "out_size" not in model_config:
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
