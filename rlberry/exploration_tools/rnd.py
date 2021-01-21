from functools import partial

import torch
import gym.spaces as spaces
from torch.nn import functional as F
from rlberry.exploration_tools.uncertainty_estimator import UncertaintyEstimator
from rlberry.exploration_tools.typing import preprocess_args
from rlberry.agents.utils.torch_models import ConvolutionalNetwork
from rlberry.agents.utils.torch_models import MultiLayerPerceptron
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
                 **kwargs):
        assert isinstance(observation_space, spaces.Box)
        UncertaintyEstimator.__init__(self, observation_space, action_space)
        self.learning_rate = learning_rate
        self.loss_fn = F.mse_loss
        self.update_period = update_period
        self.embedding_dim = embedding_dim
        self.net_fn = load(net_fn) if isinstance(net_fn, str) else \
            net_fn or partial(get_network, shape=observation_space.shape, embedding_dim=embedding_dim)
        self.net_kwargs = net_kwargs or {}
        self.device = choose_device(device)
        self.rate_power = rate_power
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

    def _get_embeddings(self, state, batch=False):
        state_tensor = state if isinstance(state, torch.Tensor) else torch.from_numpy(state)
        state_tensor = state_tensor.to(self.device)
        if not batch:
            state_tensor = state_tensor.unsqueeze(0)
        random_embedding = self.random_target_network(state_tensor)
        predicted_embedding = self.predictor_network.forward(state_tensor)
        return random_embedding, predicted_embedding

    @preprocess_args(expected_type='torch')
    def update(self,
               state,
               action=None,
               next_state=None,
               reward=None,
               **kwargs):
        random_embedding, predicted_embedding \
            = self._get_embeddings(state)

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
    def measure(self, state, actions=None, **kwargs):
        random_embedding, predicted_embedding = self._get_embeddings(state, batch=False)
        error = torch.norm(predicted_embedding.detach() - random_embedding.detach(), p=2, dim=1)
        return error.pow(2 * self.rate_power).item()

    @preprocess_args(expected_type='torch')
    def measure_batch(self, states, actions, **kwargs):
        random_embedding, predicted_embedding = self._get_embeddings(states, batch=True)
        error = torch.norm(predicted_embedding.detach() - random_embedding.detach(), p=2, dim=1)
        return error.pow(2 * self.rate_power)


class RandomNetworkDistillationWithAction(UncertaintyEstimator):
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
                 **kwargs):
        assert isinstance(observation_space, spaces.Box)
        UncertaintyEstimator.__init__(self, observation_space, action_space)
        self.learning_rate = learning_rate
        self.loss_fn = F.mse_loss
        self.update_period = update_period
        self.embedding_dim = embedding_dim
        self.net_fn = load(net_fn) if isinstance(net_fn, str) else \
            net_fn or partial(get_network, shape=(observation_space.shape[0],
                                                  observation_space.shape[1],
                                                  observation_space.shape[-1]+action_space.n) , embedding_dim=embedding_dim)
        self.net_kwargs = net_kwargs or {}
        self.device = choose_device(device)
        self.rate_power = rate_power
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

    def _get_embeddings(self, state, action, batch=False):
        state_tensor = state if isinstance(state, torch.Tensor) else torch.from_numpy(state)
        state_tensor = state_tensor.to(self.device)
        action_one_hot = torch.zeros((self.action_space.n))
        action_one_hot[action] = 1.0
        action_tensor = action_one_hot.to(self.device)
        if not batch:
            state_tensor = state_tensor.unsqueeze(0)
            action_tensor = action_tensor.unsqueeze(0)
        state_action_tensor = torch.cat((state_tensor, action_tensor.unsqueeze(1).unsqueeze(1).repeat(1, 84, 84, 1)), -1)
        random_embedding = self.random_target_network(state_action_tensor)
        predicted_embedding = self.predictor_network.forward(state_action_tensor)
        return random_embedding, predicted_embedding

    @preprocess_args(expected_type='torch')
    def update(self,
               state,
               action,
               next_state=None,
               reward=None,
               **kwargs):
        random_embedding, predicted_embedding \
            = self._get_embeddings(state, action)

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
    def measure(self, state, actions, **kwargs):
        random_embedding, predicted_embedding = self._get_embeddings(state, actions, batch=False)
        error = torch.norm(predicted_embedding.detach() - random_embedding.detach(), p=2, dim=1)
        return error.pow(2 * self.rate_power).item()

    @preprocess_args(expected_type='torch')
    def measure_batch(self, states, actions, **kwargs):
        random_embedding, predicted_embedding = self._get_embeddings(states, actions, batch=True)
        error = torch.norm(predicted_embedding.detach() - random_embedding.detach(), p=2, dim=1)
        return error.pow(2 * self.rate_power)
