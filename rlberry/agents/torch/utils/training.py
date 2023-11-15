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


def model_factory_from_env(
    env, type="MultiLayerPerceptron", net=None, filename=None, **net_kwargs
):
    """Returns a torch module after setting up input/output dimensions according to an env.

    Parameters
    ----------
    env: gym.Env
        Environment
    type: {"MultiLayerPerceptron",
           "ConvolutionalNetwork",
           "DuelingNetwork",
           "Table"}, default = "MultiLayerPerceptron"
        Type of neural network.
    net: torch.nn.Module or None
        If not None, return this neural network. It can be used to pass user-defined neural network.
    filename: str or None
        The path to a saved module or its 'state_dict'. If not None, it will load a net or a checkpoint.
    **kwargs: Dict
        Parameters to be updated, used to call :func:`~rlberry.agents.torch.utils.training.model_factory`.
    """

    if filename is not None:
        load_dict = load_from_file(filename)
        if load_dict["model"] is not None:
            net = load_dict["model"]
        checkpoint = load_dict["checkpoint"]
    else:
        checkpoint = None

    kwargs = size_model_config(env, type, **net_kwargs)

    if net is not None:
        check_network(env, net, **kwargs)

    return model_factory(type, net, checkpoint=checkpoint, **kwargs)


def load_from_file(filename):
    """Load a module or a checkpoint.

    Parameters
    ----------
    filename: str
        The path to a saved module or its 'state_dict'. It will load a net or a checkpoint.
    """
    output_dict = dict(model=None, checkpoint=None)

    loaded = torch.load(filename)
    if isinstance(loaded, torch.nn.Module):
        output_dict["model"] = loaded
    elif isinstance(loaded, dict):
        output_dict["checkpoint"] = loaded
    else:
        raise ValueError(
            "Invalid 'load_from_file'. File is expected to store either an entire model or its 'state_dict'."
        )
    return output_dict


def model_factory(
    type="MultiLayerPerceptron", net=None, filename=None, checkpoint=None, **net_kwargs
) -> nn.Module:
    """Build a neural net of a given type.

    Parameters
    ----------
    type: {"MultiLayerPerceptron",
           "ConvolutionalNetwork",
           "DuelingNetwork",
           "Table"}, default = "MultiLayerPerceptron"
        Type of neural network.
    net: torch.nn.Module or None
        If not None, return this neural network. It can be used to pass user-defined neural network.
    filename: str or None
        The path to a saved module or its 'state_dict'. If not None, it will load a net or a checkpoint.
    checkpoint: dict or None
        If not None, then it is treated as a 'state_dict' that is assigned to a neural network model.
    **net_kwargs: dict
        Parameters that vary according to each neural net type, see

        * :class:`~rlberry.agents.torch.utils.models.MultiLayerPerceptron`

        * :class:`~rlberry.agents.torch.utils.models.ConvolutionalNetwork`

        * :class:`~rlberry.agents.torch.utils.models.DuelingNetwork`

        * :class:`~rlberry.agents.torch.utils.models.Table`
    """
    from rlberry.agents.torch.utils.models import (
        MultiLayerPerceptron,
        DuelingNetwork,
        ConvolutionalNetwork,
        Table,
    )

    if filename is not None:
        load_dict = load_from_file(filename)
        if load_dict["model"] is not None:
            return load_dict["model"]
        else:
            checkpoint = load_dict["checkpoint"]

    if net is not None:
        model = net
    else:
        if type == "MultiLayerPerceptron":
            model = MultiLayerPerceptron(**net_kwargs)
        elif type == "DuelingNetwork":
            model = DuelingNetwork(**net_kwargs)
        elif type == "ConvolutionalNetwork":
            model = ConvolutionalNetwork(**net_kwargs)
        elif type == "Table":
            model = Table(**net_kwargs)
        else:
            raise ValueError("Unknown model type")

    if checkpoint is not None:
        model.load_state_dict(checkpoint)

    return model


def check_network(env, net, **model_config):
    """
    Check the neural network that it satisfies the environment and predefined model_config. If the network is not good, it should raise an error.

    Parameters
    ----------
    env : gym.Env
        An environment.
    net: torch.nn.Module
        A neural network.
    model_config : dict
        Desired parameters.
    """

    if isinstance(env.observation_space, spaces.Box):
        obs_shape = env.observation_space.shape
    else:
        raise NotImplementedError
    # elif isinstance(env.observation_space, spaces.Tuple):
    #     obs_shape = env.observation_space.spaces[0].shape
    # elif isinstance(env.observation_space, spaces.Discrete):
    #     return model_config

    if net is not None:
        # check that it is compliant with environment
        # input check
        fake_input = torch.zeros(1, *obs_shape)
        try:
            output = net(fake_input)
        except Exception as err:
            print(
                f"NN input is not compatible with the environment. Got an error {err=}, {type(err)=}"
            )
            raise
        # output check
        if "is_policy" in model_config:
            is_policy = model_config["is_policy"]
            if is_policy:
                assert isinstance(
                    output, torch.distributions.distribution.Distribution
                ), "Policy should return distribution over actions"
        else:
            if "out_size" in model_config:
                out_size = [model_config["out_size"]]
            else:
                if isinstance(env.action_space, spaces.Discrete):
                    out_size = [env.action_space.n]
                elif isinstance(env.action_space, spaces.Tuple):
                    out_size = [env.action_space.spaces[0].n]
                elif isinstance(env.action_space, spaces.Box):
                    out_size = env.action_space.shape
            assert output.shape == (
                1,
                *out_size,
            ), f"Output should be of size {out_size}"


def size_model_config(env, type=None, **model_config):
    """
    Setup input/output dimensions for the configuration of
    a model depending on the environment observation/action spaces.

    Parameters
    ----------
    env : gym.Env
        An environment.
    type: str or None
        Make configs corresponding to the chosen type of neural network.
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
    if type == "ConvolutionalNetwork":
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
