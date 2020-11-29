import logging
import torch
from gym import spaces

from rlberry.agents.utils.torch_models import size_model_config, model_factory
from rlberry.agents.utils.torch_models import trainable_parameters
from rlberry.agents.utils.torch_training import loss_function_factory
from rlberry.agents.utils.torch_training import optimizer_factory
from rlberry.agents.dqn.abstract import AbstractDQNAgent
from rlberry.agents.utils.memories import Transition
from rlberry.utils.torch import choose_device

logger = logging.getLogger(__name__)


def default_qvalue_net_fn(env):
    """
    Returns a default Q value network.
    """
    model_config = {"type": "DuelingNetwork"}
    model_config = size_model_config(env, **model_config)
    return model_factory(**model_config)


class DQNAgent(AbstractDQNAgent):
    """
    Deep Q Learning Agent.

    Parameters
    ----------
    env: gym.Env
        Environment
    n_episodes : int
        Number of episodes to train the algorithm
    horizon : int
        Maximum lenght of an episode.
    gamma : double
        Discount factor
    qvalue_net_fn : function
        Function that returns an instance of a network representing
        the Q function.
        If none, a default network is used.
    loss_function : str
        Type of loss function. Possibilities: 'l2', 'l1', 'smooth_l1'
    batch_size : int
        Batch size
    device : str
        Device used by pytorch.
    target_update : int
        Number of steps to wait before updating the target network.
    double : bool
        If true, use double Q-learning.
    optimizer_kwargs: dict
        Parameters of the optimization algorithm.
    exploration_kwargs : dict
        Parameters of exploration policy.
    memory_kwargs : dict
        Parameters of the replay buffer: capacity (int) and n_steps (int)
    """
    def __init__(self,
                 env,
                 n_episodes=1000,
                 horizon=256,
                 gamma=0.99,
                 qvalue_net_fn=None,
                 loss_function="l2",
                 batch_size=100,
                 device="cuda:best",
                 target_update=1,
                 double=True,
                 optimizer_kwargs=None,
                 exploration_kwargs=None,
                 memory_kwargs=None,
                 **kwargs):
        # Wrap arguments and initialize base class
        memory_kwargs = memory_kwargs or {}
        memory_kwargs['gamma'] = gamma
        base_args = (env, horizon, exploration_kwargs, memory_kwargs,
                     n_episodes, batch_size, target_update, double)
        AbstractDQNAgent.__init__(self, *base_args, **kwargs)

        # init
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.device = device
        self.loss_function = loss_function
        self.gamma = gamma
        #
        qvalue_net_fn = qvalue_net_fn \
            or (lambda: default_qvalue_net_fn(self.env))
        self.value_net = qvalue_net_fn()
        self.target_net = qvalue_net_fn()
        #
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        logger.debug("Number of trainable parameters: {}"
                     .format(trainable_parameters(self.value_net)))
        self.device = choose_device(self.device)
        self.value_net.to(self.device)
        self.target_net.to(self.device)
        self.loss_function = loss_function_factory(self.loss_function)
        self.optimizer = optimizer_factory(self.value_net.parameters(),
                                           **self.optimizer_kwargs)
        self.steps = 0

    def step_optimizer(self, loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def compute_bellman_residual(self, batch, target_state_action_value=None):
        # Compute concatenate the batch elements
        if not isinstance(batch.state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(tuple(torch.tensor([batch.state],
                              dtype=torch.float))).to(self.device)
            action = torch.tensor(batch.action,
                                  dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward,
                                  dtype=torch.float).to(self.device)
            next_state = torch.cat(tuple(torch.tensor([batch.next_state],
                                   dtype=torch.float))).to(self.device)
            terminal = torch.tensor(batch.terminal,
                                    dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward,
                               next_state, terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.value_net(batch.state)
        state_action_values = \
            state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                next_state_values = \
                    torch.zeros(batch.reward.shape).to(self.device)
                if self.double:
                    # Double Q-learning: pick best actions from policy network
                    _, best_actions = self.value_net(batch.next_state).max(1)
                    # Double Q-learning: estimate action values
                    # from target network
                    best_values = self.target_net(
                                    batch.next_state
                                    ).gather(1, best_actions.unsqueeze(1))\
                                     .squeeze(1)
                else:
                    best_values, _ = self.target_net(batch.next_state).max(1)
                next_state_values[~batch.terminal] \
                    = best_values[~batch.terminal]
                # Compute the expected Q values
                target_state_action_value = batch.reward \
                    + self.gamma * next_state_values

        # Compute loss
        loss = self.loss_function(state_action_values,
                                  target_state_action_value)
        return loss, target_state_action_value, batch

    def get_batch_state_values(self, states):
        values, actions = self.value_net(torch.tensor(states,
                                         dtype=torch.float)
                                         .to(self.device)).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        return self.value_net(torch.tensor(states,
                              dtype=torch.float)
                              .to(self.device)).data.cpu().numpy()

    def save(self, filename, **kwargs):
        state = {'state_dict': self.value_net.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)
        return filename

    def load(self, filename, **kwargs):
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return filename

    def initialize_model(self):
        self.value_net.reset()

    def set_writer(self, writer):
        super().set_writer(writer)
        obs_shape = self.env.observation_space.shape \
            if isinstance(self.env.observation_space, spaces.Box) else \
            self.env.observation_space.spaces[0].shape
        model_input = torch.zeros((1, *obs_shape), dtype=torch.float,
                                  device=self.device)
        self.writer.add_graph(self.value_net, input_to_model=(model_input,))
        self.writer.add_scalar("agent/trainable_parameters",
                               trainable_parameters(self.value_net), 0)
