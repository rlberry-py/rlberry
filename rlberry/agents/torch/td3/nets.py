import torch
import torch.nn as nn
from typing import Sequence, Tuple


class MLP(nn.Module):
    """MLP."""
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int], activate_last=False):
        super(MLP, self).__init__()
        hidden_sizes = tuple(hidden_sizes) + (output_dim,)
        n_layers = len(hidden_sizes)
        prev_dim = input_dim
        layers = []
        for ii in range(n_layers):
            dim = hidden_sizes[ii]
            layers.append(nn.Linear(in_features=prev_dim, out_features=dim))
            if ii < n_layers - 1 or activate_last:
                layers.append(nn.ReLU())
            prev_dim = dim
        self._mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self._mlp(x)


class TD3MLPActor(nn.Module):
    """MLP actor net for TD3.

    Parameters
    ----------
    state_dim : int
        Dimension of states
    action_dim : int
        Dimension of actions
    hidden_sizes : Tuple[int, ...]
        Number of hidden layers in the MLP.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        ):
        super(TD3MLPActor, self).__init__()
        self._mlp = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes
        )

    def forward(self, states):
        out = self._mlp(states)
        return out


class TD3MLPCritic(nn.Module):
    """MLP critic net for TD3.

    Parameters
    ----------
    state_dim : int
        Dimension of states
    action_dim : int
        Dimension of actions
    hidden_sizes : Tuple[int, ...]
        Number of hidden layers in each network. 
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        ):
        super(TD3MLPCritic, self).__init__()
        n_heads = 2
        heads_input_dim = state_dim + action_dim
        self._shared_mlp = None
        self._heads = nn.ModuleList([
            MLP(
                input_dim=heads_input_dim,
                output_dim=1,
                hidden_sizes=hidden_sizes,
                activate_last=False)
            for _ in range(n_heads)])

    def forward(self, states, actions):
        x = torch.cat((states, actions), dim=-1)
        if self._shared_mlp is not None:
            x = self._shared_mlp(x)
        head_outputs = [head(x) for head in self._heads]
        out = torch.cat(head_outputs, dim=-1)
        return out