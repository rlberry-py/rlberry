"""
Haiku modules for simple neural networks.
"""
import haiku as hk
from typing import Tuple


class MLPQNetwork(hk.Module):
    """
    MLP for Q functions with discrete number of actions.

    Parameters
    ----------
    num_actions : int
        Number of actions.
    hidden_sizes : Tuple[int, ...]
        Number of hidden layers in the MLP.
    name : str
        Identifier of the module.
    """

    def __init__(
            self,
            num_actions: int,
            hidden_sizes: Tuple[int, ...],
            name: str = 'MLPQNetwork'
    ):
        super().__init__(name=name)
        self._mlp = hk.nets.MLP(output_sizes=hidden_sizes + (num_actions,))

    def __call__(self, observation):
        out = self._mlp(observation)
        return out
