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

class MLPEnsemble(nn.Module):
    """MLP ensemble.

    Parameters
    ----------
    input_dim : int
    output_dim : int
    hidden_sizes : Tuple[int, ...]
        Number of hidden layers in the MLP.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_heads: int = 2,
        shared_hidden_sizes: Tuple[int, ...] = (),
        head_hidden_sizes: Tuple[int, ...] = (64, 64),
        ):
        super(MLPEnsemble, self).__init__()
        if len(shared_hidden_sizes) > 0:
            self._shared_mlp = MLP(
                input_dim=input_dim,
                output_dim=shared_hidden_sizes[-1],
                hidden_sizes=shared_hidden_sizes[:-1],
                activate_last=True,
            )
            heads_input_dim = shared_hidden_sizes[-1]
        else:
            heads_input_dim = input_dim
            self._shared_mlp = None
    
        self._heads = nn.ModuleList([
            MLP(
                input_dim=heads_input_dim,
                output_dim=output_dim,
                hidden_sizes=head_hidden_sizes,
                activate_last=False)
            for _ in range(n_heads)])

    def forward(self, states):
        x = states
        if self._shared_mlp is not None:
            x = self._shared_mlp(x)
        head_outputs = [torch.unsqueeze(head(x), dim=-1) for head in self._heads]
        out = torch.cat(head_outputs, dim=-1)
        return out


class ED3MLPActor(nn.Module):
    """MLP actor net for ED3.

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
        super(ED3MLPActor, self).__init__()
        self._mlp = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes
        )

    def forward(self, states):
        out = self._mlp(states)
        return out


class ED3MLPCritic(nn.Module):
    """MLP critic net for ED3.

    Parameters
    ----------
    state_dim : int
        Dimension of states
    action_dim : int
        Dimension of actions
    n_heads : int
        Number of heads
    shared_hidden_sizes : Tuple[int, ...]
        Number of hidden layers in the shared MLP.
        of the shared representation).
    head_hidden_sizes : Tuple[int, ...]
        Number of hidden layers in each head. 
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_heads: int = 4,
        shared_hidden_sizes: Tuple[int, ...] = (),
        head_hidden_sizes: Tuple[int, ...] = (64, 64),
        ):
        super(ED3MLPCritic, self).__init__()
        if len(shared_hidden_sizes) > 0:
            self._shared_mlp = MLP(
                input_dim=state_dim + action_dim,
                output_dim=shared_hidden_sizes[-1],
                hidden_sizes=shared_hidden_sizes[:-1],
                activate_last=True,
            )
            heads_input_dim = shared_hidden_sizes[-1]
        else:
            heads_input_dim = state_dim + action_dim
            self._shared_mlp = None
    
        self._heads = nn.ModuleList([
            MLP(
                input_dim=heads_input_dim,
                output_dim=1,
                hidden_sizes=head_hidden_sizes,
                activate_last=False)
            for _ in range(n_heads)])

    def forward(self, states, actions):
        x = torch.cat((states, actions), dim=-1)
        if self._shared_mlp is not None:
            x = self._shared_mlp(x)
        head_outputs = [head(x) for head in self._heads]
        out = torch.cat(head_outputs, dim=-1)
        return out