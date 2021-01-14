#
# Simple MLP and CNN models
#


import torch
import torch.nn as nn
from torch.distributions import Categorical
from gym import spaces

#
# Utility functions
#
from rlberry.agents.utils.torch_training import model_factory, activation_factory


def default_policy_net_fn(env):
    """
    Returns a default value network.
    """
    if isinstance(env.observation_space, spaces.Box):
        obs_shape = env.observation_space.shape
    elif isinstance(env.observation_space, spaces.Tuple):
        obs_shape = env.observation_space.spaces[0].shape
    else:
        raise ValueError("Incompatible observation space: {}".format(env.observation_space))
    # Assume CHW observation space
    if len(obs_shape) == 3:
        model_config = {"type": "PolicyConvolutionalNetwork", "in_channels": int(obs_shape[0]),
                        "in_height": int(obs_shape[1]),
                        "in_width": int(obs_shape[2])}
    elif len(obs_shape) == 2:
        model_config = {"type": "PolicyConvolutionalNetwork", "in_channels": int(1),
                        "in_height": int(obs_shape[0]),
                        "in_width": int(obs_shape[1])}
    elif len(obs_shape) == 1:
        model_config = {"type": "MultiLayerPerceptron", "in_size": int(obs_shape[0]),
                        "layer_sizes": [64, 64], "reshape": False, "is_policy": True}
    else:
        raise ValueError("Incompatible observation shape: {}".format(env.observation_space.shape))

    if isinstance(env.action_space, spaces.Discrete):
        model_config["out_size"] = env.action_space.n
    elif isinstance(env.action_space, spaces.Tuple):
        model_config["out_size"] = env.action_space.spaces[0].n

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
        raise ValueError("Incompatible observation space: {}".format(env.observation_space))
    # Assume CHW observation space
    if len(obs_shape) == 3:
        model_config = {"type": "ConvolutionalNetwork", "in_channels": int(obs_shape[0]),
                        "in_height": int(obs_shape[1]),
                        "in_width": int(obs_shape[2])}
    elif len(obs_shape) == 2:
        model_config = {"type": "ConvolutionalNetwork", "in_channels": int(1),
                        "in_height": int(obs_shape[0]),
                        "in_width": int(obs_shape[1])}
    elif len(obs_shape) == 1:
        model_config = {"type": "MultiLayerPerceptron", "in_size": int(obs_shape[0]),
                        "layer_sizes": [64, 64]}
    else:
        raise ValueError("Incompatible observation shape: {}".format(env.observation_space.shape))

    model_config["out_size"] = 1

    return model_factory(**model_config)


#
# Classes
#

class PolicyConvolutionalNetwork(nn.Module):
    def __init__(self,
                 activation="RELU",
                 in_channels=None,
                 in_height=None,
                 in_width=None,
                 head_mlp_kwargs=None,
                 out_size=None,
                 **kwargs):
        super().__init__()
        self.activation = activation_factory(activation)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=-1)

        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(in_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(in_height)))
        assert convh > 0 and convw > 0
        self.head_mlp_kwargs = head_mlp_kwargs or {}
        self.head_mlp_kwargs["in_size"] = convw * convh * 64
        self.head_mlp_kwargs["out_size"] = out_size
        self.head = model_factory(**self.head_mlp_kwargs)

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(torch.squeeze(x))))
        x = self.activation((self.conv2(x)))
        x = self.activation((self.conv3(x)))
        action_probs = self.softmax(self.head(x))
        dist = Categorical(action_probs)
        return dist

    def action_scores(self, x):
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))
        x = self.activation((self.conv3(x)))
        action_scores = self.head(x)
        return action_scores


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class BaseModule(torch.nn.Module):
    """
    Base torch.nn.Module implementing basic features:
        - initialization factory
        - normalization parameters
    """

    def __init__(self, activation_type="RELU", reset_type="XAVIER"):
        super().__init__()
        self.activation = activation_factory(activation_type)
        self.reset_type = reset_type

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            if self.reset_type == "XAVIER":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "ZEROS":
                torch.nn.init.constant_(m.weight.data, 0.)
            else:
                raise ValueError("Unknown reset type")
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.)

    def reset(self):
        self.apply(self._init_weights)


class Table(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.policy = nn.Embedding.from_pretrained(torch.zeros(state_size, action_size), freeze=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        action_probs = self.softmax(self.action_scores(x))
        return Categorical(action_probs)

    def action_scores(self, x):
        return self.policy(x.long())


class MultiLayerPerceptron(BaseModule):
    def __init__(self,
                 in_size=None,
                 layer_sizes=None,
                 reshape=True,
                 out_size=None,
                 activation="RELU",
                 is_policy=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.reshape = reshape
        self.layer_sizes = layer_sizes or [64, 64]
        self.out_size = out_size
        self.activation = activation_factory(activation)
        self.is_policy = is_policy
        self.softmax = nn.Softmax(dim=-1)
        sizes = [in_size] + self.layer_sizes
        layers_list = [nn.Linear(sizes[i], sizes[i + 1])
                       for i in range(len(sizes) - 1)]
        self.layers = nn.ModuleList(layers_list)
        if out_size:
            self.predict = nn.Linear(sizes[-1], out_size)

    def forward(self, x):
        if self.reshape:
            x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
        for layer in self.layers:
            x = self.activation(layer(x.float()))
        if self.out_size:
            x = self.predict(x)
        if self.is_policy:
            action_probs = self.softmax(x)
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
    def __init__(self,
                 in_size=None,
                 base_module_kwargs=None,
                 value_kwargs=None,
                 advantage_kwargs=None,
                 out_size=None):
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
        return value + advantage \
               - advantage.mean(1).unsqueeze(1).expand(-1, self.out_size)


class ConvolutionalNetwork(nn.Module):
    def __init__(self,
                 activation="RELU",
                 in_channels=None,
                 in_height=None,
                 in_width=None,
                 head_mlp_kwargs=None,
                 out_size=None,
                 **kwargs):
        super().__init__()
        self.activation = activation_factory(activation)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)

        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(in_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(in_height)))
        assert convh > 0 and convw > 0
        self.head_mlp_kwargs = head_mlp_kwargs or {}
        self.head_mlp_kwargs["in_size"] = convw * convh * 64
        self.head_mlp_kwargs["out_size"] = out_size
        self.head = model_factory(**self.head_mlp_kwargs)

    def forward(self, x):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        x = self.activation((self.conv1(x)))
        x = self.activation((self.conv2(x)))
        x = self.activation((self.conv3(x)))
        return self.head(x)


