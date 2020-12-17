#
# Attention models
#
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from rlberry.agents.utils.torch_models import BaseModule
from rlberry.agents.utils.torch_training import model_factory


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

        self.value_all = nn.Linear(self.feature_size,
                                   self.feature_size,
                                   bias=False)
        self.key_all = nn.Linear(self.feature_size,
                                 self.feature_size,
                                 bias=False)
        self.query_ego = nn.Linear(self.feature_size,
                                   self.feature_size,
                                   bias=False)
        self.attention_combine = nn.Linear(self.feature_size,
                                           self.feature_size,
                                           bias=False)

    @classmethod
    def default_config(cls):
        return {
        }

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1,
                               self.feature_size), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size,
                                               n_entities,
                                               self.heads,
                                               self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size,
                                                   n_entities,
                                                   self.heads,
                                                   self.features_per_head)
        query_ego = self.query_ego(ego).view(batch_size, 1,
                                             self.heads,
                                             self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1,
                              n_entities)).repeat((1, self.heads, 1, 1))
        value, attention_matrix = attention(query_ego,
                                            key_all,
                                            value_all,
                                            mask,
                                            nn.Dropout(self.dropout_factor))
        result = (self.attention_combine(
                    value.reshape((batch_size,
                                   self.feature_size))) + ego.squeeze(1))/2
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

        self.value_all = nn.Linear(self.feature_size,
                                   self.feature_size,
                                   bias=False)
        self.key_all = nn.Linear(self.feature_size,
                                 self.feature_size,
                                 bias=False)
        self.query_all = nn.Linear(self.feature_size,
                                   self.feature_size,
                                   bias=False)
        self.attention_combine = nn.Linear(self.feature_size,
                                           self.feature_size,
                                           bias=False)

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1,
                                        self.feature_size),
                               others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size, n_entities,
                                               self.heads,
                                               self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities,
                                                   self.heads,
                                                   self.features_per_head)
        query_all = self.query_all(input_all).view(batch_size,
                                                   n_entities,
                                                   self.heads,
                                                   self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_all = query_all.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1,
                              n_entities)).repeat((1, self.heads, 1, 1))
        value, attention_matrix = attention(query_all, key_all, value_all,
                                            mask,
                                            nn.Dropout(self.dropout_factor))
        result = (self.attention_combine(
            value.reshape((batch_size, n_entities, self.feature_size)))
            + input_all)/2
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
        self.ego_embedding = model_factory(**embedding_layer_kwargs)
        self.embedding = model_factory(**embedding_layer_kwargs)

        attention_layer_kwargs = attention_layer_kwargs or {}
        self.attention_layer = EgoAttention(**attention_layer_kwargs)

        output_layer_kwargs = output_layer_kwargs or {}
        output_layer_kwargs["in_size"] = self.attention_layer.feature_size
        output_layer_kwargs["out_size"] = self.out_size
        self.output_layer = model_factory(**output_layer_kwargs)

    def forward(self, x):
        ego_embedded_att, _ = self.forward_attention(x)
        return self.output_layer(ego_embedded_att)

    def split_input(self, x, mask=None):
        # Dims: batch, entities, features
        if len(x.shape) == 2:
            x = x.unsqueeze(axis=0)
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            aux = self.presence_feature_idx
            mask = x[:, :, aux:aux + 1] < 0.5
        return ego, others, mask

    def forward_attention(self, x):
        ego, others, mask = self.split_input(x)
        ego = self.ego_embedding(ego)
        others = self.embedding(others)
        return self.attention_layer(ego, others, mask)

    def get_attention_matrix(self, x):
        _, attention_matrix = self.forward_attention(x)
        return attention_matrix

    def action_scores(self, x):
        ego_embedded_att, _ = self.forward_attention(x)
        return self.output_layer.action_scores(ego_embedded_att)


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
