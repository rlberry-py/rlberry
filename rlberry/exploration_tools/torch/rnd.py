from functools import partial

import torch
import gym.spaces as spaces
from torch.nn import functional as F

from rlberry.agents.utils.memories import ReplayMemory
from rlberry.exploration_tools.uncertainty_estimator import UncertaintyEstimator
from rlberry.exploration_tools.typing import preprocess_args
from rlberry.agents.torch.utils.models import ConvolutionalNetwork
from rlberry.agents.torch.utils.models import MultiLayerPerceptron
from rlberry.utils.factory import load
from rlberry.utils.torch import choose_device


def get_network(shape, embedding_dim):
    if len(shape) == 3:
        if shape[2] < shape[0] and shape[2] < shape[1]:
            W, H, C = shape
            transpose_obs = True
        elif shape[0] < shape[1] and shape[0] < shape[2]:
            C, H, W = shape
            transpose_obs = False
        else:
            raise ValueError("Unknown image convention")

        return ConvolutionalNetwork(in_channels=C,
                                    in_width=W,
                                    in_height=H,
                                    out_size=embedding_dim,
                                    activation="ELU",
                                    transpose_obs=transpose_obs,
                                    is_policy=False)
    elif len(shape) == 2:
        H, W = shape
        return ConvolutionalNetwork(in_channels=1,
                                    in_width=W,
                                    in_height=H,
                                    activation="ELU",
                                    out_size=embedding_dim)

    elif len(shape) == 1:
        return MultiLayerPerceptron(in_size=shape[0],
                                    activation="RELU",
                                    layer_sizes=[64, 64],
                                    out_size=embedding_dim)
    else:
        raise ValueError("Incompatible observation shape: {}"
                         .format(shape))


class RandomNetworkDistillation(UncertaintyEstimator):
    """
    References
    ----------
    Burda Yuri, Harrison Edwards, Amos Storkey, and Oleg Klimov. 2018.
    "Exploration by random network distillation."
    In International Conference on Learning Representations.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 learning_rate=0.001,
                 update_period=100,
                 embedding_dim=10,
                 net_fn=None,
                 net_kwargs=None,
                 device="cuda:best",
                 rate_power=0.5,
                 batch_size=10,
                 memory_size=10000,
                 with_action=False,
                 **kwargs):
        assert isinstance(observation_space, spaces.Box)
        UncertaintyEstimator.__init__(self, observation_space, action_space)
        self.learning_rate = learning_rate
        self.loss_fn = F.mse_loss
        self.update_period = update_period
        self.embedding_dim = embedding_dim
        out_size = embedding_dim * action_space.n if with_action else embedding_dim
        self.net_fn = load(net_fn) if isinstance(net_fn, str) else \
            net_fn or partial(get_network, shape=observation_space.shape, embedding_dim=out_size)
        self.net_kwargs = net_kwargs or {}
        if "out_size" in self.net_kwargs:
            self.net_kwargs["out_size"] = out_size
        self.device = choose_device(device)
        self.rate_power = rate_power
        self.batch_size = batch_size
        self.memory = ReplayMemory(capacity=memory_size)
        self.with_action = with_action
        self.reset()

    def reset(self, **kwargs):
        self.random_target_network = self.net_fn(**self.net_kwargs).to(self.device)
        self.predictor_network = self.net_fn(**self.net_kwargs).to(self.device)
        self.rnd_optimizer = torch.optim.Adam(
            self.predictor_network.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999))

        self.count = 0
        self.loss = torch.tensor(0.0).to(self.device)

    def _get_embeddings(self, state, action=None, batch=False, all_actions=False):
        state = state.to(self.device)
        if not batch:
            state = state.unsqueeze(0)

        random_embedding = self.random_target_network(state)
        predicted_embedding = self.predictor_network(state)

        if self.with_action:
            random_embedding = random_embedding.view((state.shape[0], self.action_space.n, -1))
            predicted_embedding = predicted_embedding.view((state.shape[0], self.action_space.n, -1))
            if not all_actions:
                action = action.long().to(self.device)
                if not batch:
                    action = action.unsqueeze(0)
                action = action.unsqueeze(1).repeat(1, random_embedding.shape[-1]).unsqueeze(1)
                random_embedding = random_embedding.gather(1, action).squeeze(1)
                predicted_embedding = predicted_embedding.gather(1, action).squeeze(1)
        return random_embedding, predicted_embedding

    @preprocess_args(expected_type='torch')
    def update(self,
               state,
               action=None,
               next_state=None,
               reward=None,
               **kwargs):

        batch = [(state, action)]
        if self.batch_size > 0 and not self.memory.is_empty():
            batch += self.memory.sample(self.batch_size)
            self.memory.push((state, action))
        states, actions = zip(*batch)
        states = torch.stack(states)
        if self.with_action:
            actions = torch.stack(actions)

        random_embedding, predicted_embedding = self._get_embeddings(states, actions, batch=True)

        self.loss += self.loss_fn(random_embedding.detach(),
                                  predicted_embedding)

        self.count += 1
        if self.count % self.update_period == 0:
            self.loss /= self.update_period
            self.rnd_optimizer.zero_grad()
            self.loss.backward()
            self.rnd_optimizer.step()
            self.loss = torch.tensor(0.0).to(self.device)

    @preprocess_args(expected_type='torch')
    def measure(self, state, action=None, **kwargs):
        random_embedding, predicted_embedding = self._get_embeddings(state, action, batch=False)
        error = torch.norm(predicted_embedding.detach() - random_embedding.detach(), p=2, dim=-1)
        return error.pow(2 * self.rate_power).item()

    @preprocess_args(expected_type='torch')
    def measure_batch(self, states, actions, **kwargs):
        random_embedding, predicted_embedding = self._get_embeddings(states, actions, batch=True)
        error = torch.norm(predicted_embedding.detach() - random_embedding.detach(), p=2, dim=-1)
        return error.pow(2 * self.rate_power)

    @preprocess_args(expected_type='torch')
    def measure_batch_all_actions(self, states, **kwargs):
        """
        Measure N(s,a) for all a in A.

        Parameters
        ----------
        states: a batch of states, of shape [B x <state_shape>]

        Returns
        -------
        N(s,a): an array of shape B x A
        """
        assert self.with_action
        random_embedding, predicted_embedding = self._get_embeddings(states, None, batch=True, all_actions=True)
        error = torch.norm(predicted_embedding.detach() - random_embedding.detach(), p=2, dim=-1)
        return error.pow(2 * self.rate_power)
