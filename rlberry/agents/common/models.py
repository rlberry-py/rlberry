import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from gym import spaces
from torch.nn import functional as F


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(ValueNet, self).__init__()
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        state_value = self.critic(state)
        return torch.squeeze(state_value)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyNet, self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        action_probs = self.softmax(self.actor(state))
        dist = Categorical(action_probs)
        return dist

    def action_scores(self, state):
        action_scores = self.actor(state)
        return action_scores


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        return action, dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


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


class MultiLayerPerceptron(BaseModule):
    def __init__(self,
                 in_size=None,
                 layer_sizes=None,
                 reshape="True",
                 out_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.reshape = reshape
        self.layer_sizes = layer_sizes or [64, 64]
        self.out_size = out_size
        sizes = [in_size] + self.layer_sizes
        layers_list = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.layers = nn.ModuleList(layers_list)
        if out_size:
            self.predict = nn.Linear(sizes[-1], out_size)

    def forward(self, x):
        if self.reshape:
            x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.out_size:
            x = self.predict(x)
        return x


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
        return value + advantage - advantage.mean(1).unsqueeze(1).expand(-1, self.out_size)


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
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(in_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(in_height)))
        assert convh > 0 and convw > 0
        head_mlp_kwargs = head_mlp_kwargs or {}
        head_mlp_kwargs["in_size"] = convw * convh * 64
        head_mlp_kwargs["out_size"] = out_size
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


class EgoAttention(BaseModule):
    def __init__(self,
                 feature_size=64,
                 heads=4,
                 dropout_factor=0):
        super().__init__()
        self.feature_size = feature_size
        self.heads = heads
        self.dropout_factor = dropout_factor
        self.features_per_head = int(self.feature_size / self.heads)

        self.value_all = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.key_all = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.query_ego = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.attention_combine = nn.Linear(self.feature_size, self.feature_size, bias=False)

    @classmethod
    def default_config(cls):
        return {
        }

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.feature_size), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size, n_entities, self.heads, self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities, self.heads, self.features_per_head)
        query_ego = self.query_ego(ego).view(batch_size, 1, self.heads, self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.heads, 1, 1))
        value, attention_matrix = attention(query_ego, key_all, value_all, mask,
                                            nn.Dropout(self.dropout_factor))
        result = (self.attention_combine(value.reshape((batch_size, self.feature_size))) + ego.squeeze(1))/2
        return result, attention_matrix


class SelfAttention(BaseModule):
    def __init__(self,
                 feature_size=64,
                 heads=4,
                 dropout_factor=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.heads = heads
        self.dropout_factor = dropout_factor
        self.features_per_head = int(self.feature_size / self.heads)

        self.value_all = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.key_all = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.query_all = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.attention_combine = nn.Linear(self.feature_size, self.feature_size, bias=False)

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.feature_size), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size, n_entities, self.heads, self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities, self.heads, self.features_per_head)
        query_all = self.query_all(input_all).view(batch_size, n_entities, self.heads, self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_all = query_all.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.heads, 1, 1))
        value, attention_matrix = attention(query_all, key_all, value_all, mask,
                                            nn.Dropout(self.dropout_factor))
        result = (self.attention_combine(value.reshape((batch_size, n_entities, self.feature_size))) + input_all)/2
        return result, attention_matrix


class EgoAttentionNetwork(BaseModule):
    def __init__(self,
                 in_size=None,
                 out_size=None,
                 presence_feature_idx=0,
                 embedding_layer_kwargs=None,
                 attention_layer_kwargs=None,
                 output_layer_kwargs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_size = out_size
        self.presence_feature_idx = presence_feature_idx
        embedding_layer_kwargs = embedding_layer_kwargs or {}
        if not embedding_layer_kwargs.get("in_size", None):
            embedding_layer_kwargs["in_size"] = in_size

        attention_layer_kwargs = attention_layer_kwargs or {}
        self.attention_layer = EgoAttention(**attention_layer_kwargs)

        output_layer_kwargs = output_layer_kwargs or {}
        output_layer_kwargs["in_size"] = self.attention_layer.feature_size
        output_layer_kwargs["out_size"] = self.out_size

        self.embedding = model_factory(self.embedding_layer)
        self.output_layer = model_factory(self.output_layer)

    def forward(self, x):
        ego_embedded_att, _ = self.forward_attention(x)
        return self.output_layer(ego_embedded_att)

    def split_input(self, x, mask=None):
        # Dims: batch, entities, features
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            mask = x[:, :, self.presence_feature_idx:self.presence_feature_idx + 1] < 0.5
        return ego, others, mask

    def forward_attention(self, x):
        ego, others, mask = self.split_input(x)
        ego, others = self.embedding(ego), self.embedding(others)
        return self.attention_layer(ego, others, mask)

    def get_attention_matrix(self, x):
        _, attention_matrix = self.forward_attention(x)
        return attention_matrix


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute a Scaled Dot Product Attention.

    Parameters
    ----------
    query
        size: batch, head, 1 (ego-entity), features
    key
        size: batch, head, entities, features
    value
        size: batch, head, entities, features
    mask
        size: batch,  head, 1 (absence feature), 1 (ego-entity)
    dropout

    Returns
    -------
    The attention softmax(QK^T/sqrt(dk))V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


def activation_factory(activation_type):
    if activation_type == "RELU":
        return F.relu
    elif activation_type == "TANH":
        return torch.tanh
    else:
        raise ValueError("Unknown activation_type: {}".format(activation_type))


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def model_factory(type="MultiLayerPerceptron", **kwargs) -> nn.Module:
    if type == "MultiLayerPerceptron":
        return MultiLayerPerceptron(**kwargs)
    elif type == "DuelingNetwork":
        return DuelingNetwork(**kwargs)
    elif type == "ConvolutionalNetwork":
        return ConvolutionalNetwork(**kwargs)
    elif type == "EgoAttentionNetwork":
        return EgoAttentionNetwork(**kwargs)
    else:
        raise ValueError("Unknown model type")
