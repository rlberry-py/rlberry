import logging
import torch
from gym import spaces

from rlberry.agents.utils.torch_training import loss_function_factory, model_factory, size_model_config, \
    trainable_parameters
from rlberry.agents.utils.torch_training import optimizer_factory
from rlberry.agents.dqn.abstract import AbstractDQNAgent
from rlberry.agents.utils.memories import Transition
from rlberry.utils.factory import load
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
    qvalue_net_fn : function(env, **kwargs)
        Function that returns an instance of a network representing
        the Q function.
        If none, a default network is used.
    qvalue_net_kwargs:
        kwargs for qvalue_net_fn
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
    learning_rate : double
        Optimizer learning rate.
    epsilon_init : double
        Initial value of epsilon in epsilon-greedy exploration
    epsilon_final : double
        Final value of epsilon in epsilon-greedy exploration
    epsilon_decay : int
        After `epsilon_decay` steps, epsilon approaches `epsilon_final`.
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    memory_capacity : int
        Capacity of the replay buffer (in number of transitions).
    use_bonus : bool, default = False
        If true, compute an 'exploration_bonus' and add it to the reward.
        See also UncertaintyEstimatorWrapper.
    uncertainty_estimator_kwargs : dict
        Arguments for the UncertaintyEstimatorWrapper
    """
    name = 'DQN'

    def __init__(self,
                 env,
                 n_episodes=1000,
                 horizon=256,
                 gamma=0.99,
                 loss_function="l2",
                 batch_size=100,
                 device="cuda:best",
                 target_update=1,
                 learning_rate=0.001,
                 epsilon_init=1.0,
                 epsilon_final=0.1,
                 epsilon_decay=5000,
                 optimizer_type='ADAM',
                 qvalue_net_fn=None,
                 qvalue_net_kwargs=None,
                 double=True,
                 memory_capacity=10000,
                 use_bonus=False,
                 uncertainty_estimator_kwargs=None,
                 **kwargs):
        # Wrap arguments and initialize base class
        memory_kwargs = {
                        'capacity': memory_capacity,
                        'n_steps': 1,
                        'gamma': gamma
                        }
        exploration_kwargs = {
                             'method': "EpsilonGreedy",
                             'temperature': epsilon_init,
                             'final_temperature': epsilon_final,
                             'tau': epsilon_decay,
                            }
        base_args = (env, horizon, exploration_kwargs, memory_kwargs,
                     n_episodes, batch_size, target_update, double,
                     use_bonus, uncertainty_estimator_kwargs)
        AbstractDQNAgent.__init__(self, *base_args, **kwargs)

        # init
        self.optimizer_kwargs = {'optimizer_type': optimizer_type,
                                 'lr': learning_rate}
        self.device = choose_device(device)
        self.loss_function = loss_function
        self.gamma = gamma
        #
        qvalue_net_kwargs = qvalue_net_kwargs or {}

        qvalue_net_fn = load(qvalue_net_fn) if isinstance(qvalue_net_fn, str) else \
            qvalue_net_fn or default_qvalue_net_fn
        self.value_net = qvalue_net_fn(self.env, **qvalue_net_kwargs)
        self.target_net = qvalue_net_fn(self.env, **qvalue_net_kwargs)
        #
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        logger.info("Number of trainable parameters: {}"
                    .format(trainable_parameters(self.value_net)))
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
        if self.writer:
            obs_shape = self.env.observation_space.shape \
                if isinstance(self.env.observation_space, spaces.Box) else \
                self.env.observation_space.spaces[0].shape
            model_input = torch.zeros((1, *obs_shape), dtype=torch.float,
                                      device=self.device)
            self.writer.add_graph(self.value_net, input_to_model=(model_input,))
            self.writer.add_scalar("agent/trainable_parameters",
                                   trainable_parameters(self.value_net), 0)

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        batch_size = trial.suggest_categorical('batch_size',
                                               [32, 64, 128, 256, 512])
        gamma = trial.suggest_categorical('gamma',
                                          [0.95, 0.99])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)

        target_update = trial.suggest_categorical('target_update',
                                                  [1, 250, 500, 1000])

        epsilon_final = trial.suggest_loguniform('epsilon_final', 1e-2, 1e-1)

        epsilon_decay = trial.suggest_categorical('target_update',
                                                  [1000, 5000, 10000])

        return {
                'batch_size': batch_size,
                'gamma': gamma,
                'learning_rate': learning_rate,
                'target_update': target_update,
                'epsilon_final': epsilon_final,
                'epsilon_decay': epsilon_decay,
                }
