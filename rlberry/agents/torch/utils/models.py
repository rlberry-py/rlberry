#
# Simple MLP and CNN models
#
from functools import partial

import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from rlberry.agents.torch.utils.training import model_factory, activation_factory


def default_twinq_net_fn(env):
    """
    Returns a default Twinq network
    """
    assert isinstance(env.action_space, spaces.Discrete)
    if isinstance(env.observation_space, spaces.Box):
        obs_shape = env.observation_space.shape
    elif isinstance(env.observation_space, spaces.Tuple):
        obs_shape = env.observation_space.spaces[0].shape
    else:
        raise ValueError(
            "Incompatible observation space: {}".format(env.observation_space)
        )
    # Assume CHW observation space

    if len(obs_shape) == 1:
        model_config = {
            "type": "MultiLayerPerceptron",
            "in_size": int(obs_shape[0]) + int(env.action_space.n),
            "layer_sizes": [64, 64],
        }
    else:
        raise ValueError(
            "Incompatible observation shape: {}".format(env.observation_space.shape)
        )

    model_config["out_size"] = 1

    q1 = model_factory(**model_config)
    q2 = model_factory(**model_config)

    return (q1, q2)


def default_policy_net_fn(env):
    """
    Returns a default policy network.
    """
    if isinstance(env.observation_space, spaces.Box):
        obs_shape = env.observation_space.shape
    elif isinstance(env.observation_space, spaces.Tuple):
        obs_shape = env.observation_space.spaces[0].shape
    else:
        raise ValueError(
            "Incompatible observation space: {}".format(env.observation_space)
        )

    if len(obs_shape) == 3:
        if obs_shape[0] < obs_shape[1] and obs_shape[0] < obs_shape[2]:
            # Assume CHW observation space
            model_config = {
                "type": "ConvolutionalNetwork",
                "is_policy": True,
                "in_channels": int(obs_shape[0]),
                "in_height": int(obs_shape[1]),
                "in_width": int(obs_shape[2]),
            }
        elif obs_shape[2] < obs_shape[0] and obs_shape[2] < obs_shape[1]:
            # Assume WHC observation space
            model_config = {
                "type": "ConvolutionalNetwork",
                "is_policy": True,
                "transpose_obs": True,
                "in_channels": int(obs_shape[2]),
                "in_height": int(obs_shape[1]),
                "in_width": int(obs_shape[0]),
            }
    elif len(obs_shape) == 2:
        model_config = {
            "type": "ConvolutionalNetwork",
            "is_policy": True,
            "in_channels": int(1),
            "in_height": int(obs_shape[0]),
            "in_width": int(obs_shape[1]),
        }
    elif len(obs_shape) == 1:
        model_config = {
            "type": "MultiLayerPerceptron",
            "in_size": int(obs_shape[0]),
            "layer_sizes": [64, 64],
            "reshape": False,
            "is_policy": True,
        }
    else:
        raise ValueError(
            "Incompatible observation shape: {}".format(env.observation_space.shape)
        )

    if isinstance(env.action_space, spaces.Discrete):
        model_config["out_size"] = env.action_space.n
        model_config["ctns_actions"] = False
    elif isinstance(env.action_space, spaces.Tuple):
        model_config["out_size"] = env.action_space.spaces[0].n
        model_config["ctns_actions"] = False
    elif isinstance(env.action_space, spaces.Box):
        model_config["out_size"] = env.action_space.shape[0]
        model_config["ctns_actions"] = True

    return model_factory(**model_config)


def default_value_net_fn(env):
    """
    Returns a default value network.
    """
    if isinstance(env.observation_space, spaces.Box):
        obs_shape = env.observation_space.shape
    elif isinstance(env.observation_space, spaces.Tuple):
        obs_shape = env.observation_space.spaces[0].shape
    else:
        raise ValueError(
            "Incompatible observation space: {}".format(env.observation_space)
        )
    # Assume CHW observation space
    if len(obs_shape) == 3:
        model_config = {
            "type": "ConvolutionalNetwork",
            "in_channels": int(obs_shape[0]),
            "in_height": int(obs_shape[1]),
            "in_width": int(obs_shape[2]),
        }
    elif len(obs_shape) == 2:
        model_config = {
            "type": "ConvolutionalNetwork",
            "in_channels": int(1),
            "in_height": int(obs_shape[0]),
            "in_width": int(obs_shape[1]),
        }
    elif len(obs_shape) == 1:
        model_config = {
            "type": "MultiLayerPerceptron",
            "in_size": int(obs_shape[0]),
            "layer_sizes": [64, 64],
        }
    else:
        raise ValueError(
            "Incompatible observation shape: {}".format(env.observation_space.shape)
        )

    model_config["out_size"] = 1

    return model_factory(**model_config)


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class BaseModule(torch.nn.Module):
    """
    Base torch.nn.Module implementing basic features:
        - initialization factory
        - normalization parameters
    """

    def __init__(self, activation_type="RELU", reset_type="xavier"):
        super().__init__()
        self.activation = activation_factory(activation_type)
        self.reset_type = reset_type

    def _init_weights(self, m, param=None):
        if hasattr(m, "weight"):
            if self.reset_type == "xavier":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "zeros":
                torch.nn.init.constant_(m.weight.data, 0.0)
            elif self.reset_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=param)
            else:
                raise ValueError("Unknown reset type")
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

    def reset(self):
        self.apply(self._init_weights)


class Table(torch.nn.Module):
    """Torch module for a policy for discrete state-action spaces.

    Parameters
    ----------
    state_size: int
        Number of states
    action_size: int
        Number of actions
    """

    def __init__(self, state_size, action_size):
        super().__init__()
        self.policy = nn.Embedding.from_pretrained(
            torch.zeros(state_size, action_size), freeze=False
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        action_probs = self.softmax(self.action_scores(x))
        return Categorical(action_probs)

    def action_scores(self, x):
        return self.policy(x.long())


class MultiLayerPerceptron(BaseModule):
    """Torch module for an MLP.

    Parameters
    ----------
    in_size: int
        Input size
    layer_sizes: Sequence[int]
        Dimensions of each hidden layer.
    reshape: bool, default = True
        If True, input tensors are reshaped to (batch_size, dim)
    out_size: int, optional
        Output size. If None, the output size is given by the last
        element of layer_sizes.
    activation: {"RELU", "TANH", "ELU"}
        Activation function.
    is_policy: bool, default=False
        If true, the :meth:`forward` method returns a distribution over the
        output.
    ctns_actions: bool, default=False
        If true, the :meth:`forward` method returns a normal distribution
        corresponding to the output. Otherwise, a categorical distribution
        is returned.
    std0: float, default=1.0
        Initial standard deviation for the normal distribution. Only used
        if ctns_actions and is_policy are True.
    reset_type: {"xavier", "orthogonal", "zeros"}, default="orthogonal"
        Type of weight initialization.
    pred_init_scale: float, default="auto"
        Scale of the initial weights of the output layer. If "auto", the
        scale is set to 0.01 for policy networks and 1.0 otherwise.
    """

    def __init__(
        self,
        in_size=None,
        layer_sizes=None,
        reshape=False,
        out_size=None,
        activation="RELU",
        is_policy=False,
        ctns_actions=False,
        std0=1.0,
        reset_type="orthogonal",
        pred_init_scale="auto",
        **kwargs
    ):
        super().__init__(reset_type=reset_type, **kwargs)

        self.reshape = reshape
        self.layer_sizes = layer_sizes or [64, 64]
        self.layer_sizes = list(self.layer_sizes)
        self.out_size = out_size
        self.activation = activation_factory(activation)
        self.is_policy = is_policy
        self.ctns_actions = ctns_actions
        self.std0 = std0
        self.pred_init_scale = pred_init_scale

        sizes = [in_size] + self.layer_sizes
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
        if out_size:
            if ctns_actions:
                self.logstd = nn.Parameter(np.log(std0) * torch.ones(out_size))
            self.predict = nn.Linear(sizes[-1], out_size)
        self.reset()

    def reset(self):
        self.apply(partial(self._init_weights, param=np.log(2)))
        if self.out_size:
            if self.pred_init_scale == "auto":
                pred_init_scale = 0.01 if self.is_policy else 1.0
            else:
                pred_init_scale = self.pred_init_scale
            self._init_weights(self.predict, param=pred_init_scale)

    def forward(self, x):
        if self.reshape:
            x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
        for layer in self.layers:
            x = self.activation(layer(x.float()))
        if self.out_size:
            x = self.predict(x)
        if self.is_policy:
            if self.ctns_actions:
                std = torch.exp(self.logstd.expand_as(x))
                dist = Normal(x, std)
            else:
                action_probs = F.softmax(x, dim=-1)
                dist = Categorical(action_probs)
            return dist
        return x

    def action_scores(self, x):
        if self.is_policy:
            if self.reshape:
                x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
            for layer in self.layers:
                x = self.activation(layer(x.float()))
            if self.out_size:
                action_scores = self.predict(x)
            return action_scores


class DuelingNetwork(BaseModule):
    """Torch module for a DQN dueling network based on a MultiLayerPerceptron.

    Parameters
    -----------
    in_size: int
        Input size
    base_module_kwargs: dict
        Parameters for :func:`~rlberry.agents.torch.utils.training.model_factory`
        to build shared (MLP) architecture for the advantage and value nets.
    value_kwargs: dict
        Parameters for :func:`~rlberry.agents.torch.utils.training.model_factory`
        to build value network (MLP).
    advantage_kwargs: dict
        Parameters for :func:`~rlberry.agents.torch.utils.training.model_factory`
        to build advantage network (MLP).
    out_size: int
        Output size.
    """

    def __init__(
        self,
        in_size=None,
        base_module_kwargs=None,
        value_kwargs=None,
        advantage_kwargs=None,
        out_size=None,
    ):
        super().__init__()
        self.out_size = out_size
        base_module_kwargs = base_module_kwargs or {}
        base_module_kwargs["in_size"] = in_size
        self.base_module = model_factory(**base_module_kwargs)
        value_kwargs = value_kwargs or {}
        value_kwargs["in_size"] = self.base_module.layer_sizes[-1]
        value_kwargs["out_size"] = 1
        self.value = model_factory(**value_kwargs)
        advantage_kwargs = advantage_kwargs or {}
        advantage_kwargs["in_size"] = self.base_module.layer_sizes[-1]
        advantage_kwargs["out_size"] = out_size
        self.advantage = model_factory(**advantage_kwargs)

    def forward(self, x):
        x = self.base_module(x)
        value = self.value(x).expand(-1, self.out_size)
        advantage = self.advantage(x)
        return (
            value + advantage - advantage.mean(1).unsqueeze(1).expand(-1, self.out_size)
        )


class ConvolutionalNetwork(nn.Module):
    """Torch module for a CNN.

    Expects inputs of shape BCHW, where
    B = batch size;
    C = number of channels;
    H = height;
    W = width.

    For the CNN forward, if the tensor has more than 4 dimensions (not BCHW), it keeps the 3 last dimension as CHW and merge all first ones into 1 (Batch). Go through the CNN + MLP, then split the first dimension as before.

    Parameters
    ----------
    activation: {"RELU", "TANH", "ELU"}
        Activation function.
    in_channels: int
        Number of input channels C
    in_height: int
        Input height H
    in_width: int
        Input width W
    head_mlp_kwargs: dict, optional
        Parameters to build an MLP
        (:class:`~rlberry.agents.torch.utils.models.MultiLayerPerceptron`)
        using the factory
        :func:`~rlberry.agents.torch.utils.training.model_factory`

    """

    def __init__(
        self,
        activation="RELU",
        in_channels=None,
        in_height=None,
        in_width=None,
        head_mlp_kwargs=None,
        out_size=None,
        is_policy=False,
        transpose_obs=False,
        **kwargs
    ):
        super().__init__()
        self.activation = activation_factory(activation)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)

        # MLP Head
        self.head_mlp_kwargs = head_mlp_kwargs or {}
        self.head_mlp_kwargs["in_size"] = self._get_conv_out_size(
            [in_channels, in_height, in_width]
        )  # Number of Linear input connections depends on output of conv layers
        self.head_mlp_kwargs["out_size"] = out_size
        self.head_mlp_kwargs["is_policy"] = is_policy
        self.head = model_factory(**self.head_mlp_kwargs)

        self.is_policy = is_policy
        self.transpose_obs = transpose_obs

    def _get_conv_out_size(self, shape):
        """
        Computes the output dimensions of the convolution network.
        Shape : dimension of the input of the CNN
        """
        conv_result = self.activation((self.conv1(torch.zeros(1, *shape))))
        conv_result = self.activation((self.conv2(conv_result)))
        conv_result = self.activation((self.conv3(conv_result)))
        return int(np.prod(conv_result.size()))

    def convolutions(self, x):
        x = x.float()
        if (
            len(x.shape) == 3
        ):  # if there is no batch (CHW), add one dimension to specify batch of 1 (and get format BCHW)
            x = x.unsqueeze(0)
        if self.transpose_obs:
            x = torch.transpose(x, -1, -3)
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))
        x = self.activation((self.conv3(x)))
        x = x.view(x.size(0), -1)  # flatten
        return x

    def forward(self, x):
        """
        Forward convolutional network

        Parameters
        ----------
        x: torch.tensor
            Tensor of shape BCHW (Batch,Chanel,Height,Width : if more than 4 dimensions, merge all the first in batch dimension)
        """
        flag_view_to_change = False

        if len(x.shape) > 4:
            flag_view_to_change = True
            dim_to_retore = x.shape[:-3]
            inputview_size = tuple((-1,)) + tuple(x.shape[-3:])
            outputview_size = tuple(dim_to_retore) + tuple(
                (self.head_mlp_kwargs["out_size"],)
            )
            x = x.view(inputview_size)

        conv_result = self.convolutions(x)
        output_result = self.head(
            conv_result.view(conv_result.size()[0], -1)
        )  # give the 'conv_result' flattenned in 2 dimensions (batch and other) to the MLP (head)

        if flag_view_to_change:
            output_result = output_result.view(outputview_size)

        return output_result

    def action_scores(self, x):
        return self.head.action_scores(self.convolutions(x))
