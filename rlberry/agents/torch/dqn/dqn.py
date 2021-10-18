import torch
from gym import spaces
import logging
import numpy as np

from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.utils.memories import Transition, PrioritizedReplayMemory, TransitionReplayMemory
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from rlberry.exploration_tools.online_discretization_counter import OnlineDiscretizationCounter
from rlberry.wrappers.uncertainty_estimator_wrapper import UncertaintyEstimatorWrapper
from rlberry.agents.torch.dqn.exploration import exploration_factory
from rlberry.agents.torch.utils.training import loss_function_factory, model_factory, size_model_config, \
    trainable_parameters, optimizer_factory
from rlberry.seeding import Seeder
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


class DQNAgent(AgentWithSimplePolicy):
    """
    Deep Q Learning Agent.

    Parameters
    ----------
    env: gym.Env
        Environment
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
    prioritized_replay: bool
        Use prioritized replay.
    """
    name = 'DQN'

    def __init__(self,
                 env,
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
                 prioritized_replay=True,
                 update_frequency=1,
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
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.use_bonus = use_bonus
        if self.use_bonus:
            self.env = UncertaintyEstimatorWrapper(
                self.env, **uncertainty_estimator_kwargs)
        self.horizon = horizon
        self.exploration_kwargs = exploration_kwargs or {}
        self.memory_kwargs = memory_kwargs or {}
        self.batch_size = batch_size
        self.target_update = target_update
        self.double = double

        assert isinstance(env.action_space, spaces.Discrete), \
            "Only compatible with Discrete action spaces."

        self.prioritized_replay = prioritized_replay
        memory_class = PrioritizedReplayMemory if prioritized_replay else TransitionReplayMemory
        self.memory = memory_class(**self.memory_kwargs)
        self.exploration_policy = \
            exploration_factory(self.env.action_space,
                                **self.exploration_kwargs)
        self.training = True
        self.steps = 0
        self.episode = 0

        self.optimizer_kwargs = {'optimizer_type': optimizer_type,
                                 'lr': learning_rate}
        self.device = choose_device(device)
        self.loss_function = loss_function
        self.gamma = gamma

        qvalue_net_kwargs = qvalue_net_kwargs or {}
        qvalue_net_fn = load(qvalue_net_fn) if isinstance(qvalue_net_fn, str) else \
            qvalue_net_fn or default_qvalue_net_fn
        self.value_net = qvalue_net_fn(self.env, **qvalue_net_kwargs)
        self.target_net = qvalue_net_fn(self.env, **qvalue_net_kwargs)

        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        logger.info("Number of trainable parameters: {}"
                    .format(trainable_parameters(self.value_net)))
        self.value_net.to(self.device)
        self.target_net.to(self.device)
        self.loss_function = loss_function_factory(self.loss_function)
        self.optimizer = optimizer_factory(self.value_net.parameters(),
                                           **self.optimizer_kwargs)
        self.update_frequency = update_frequency
        self.steps = 0

    def fit(self, budget: int, **kwargs):
        del kwargs
        for self.episode in range(budget):
            if self.writer:
                state = self.env.reset()
                values = self.get_state_action_values(state)
                for i, value in enumerate(values):
                    self.writer.add_scalar(f"agent/action_value_{i}", value, self.episode)
            total_reward, total_bonus, total_success, length = self._run_episode()
            if self.episode % 20 == 0:
                logger.info(f"Episode {self.episode + 1}/{budget}, total reward {total_reward}")
            if self.writer:
                self.writer.add_scalar("episode_rewards", total_reward, self.episode)
                self.writer.add_scalar("episode/total_reward", total_reward, self.episode)
                self.writer.add_scalar("episode/total_bonus", total_bonus, self.episode)
                self.writer.add_scalar("episode/total_success", total_success, self.episode)
                self.writer.add_scalar("episode/length", length, self.episode)
                if self.use_bonus and \
                        (isinstance(self.env.uncertainty_estimator, OnlineDiscretizationCounter) or
                         isinstance(self.env.uncertainty_estimator, DiscreteCounter)):
                    n_visited_states = (self.env.uncertainty_estimator.N_sa.sum(axis=1) > 0).sum()
                    self.writer.add_scalar("debug/n_visited_states", n_visited_states, self.episode)

    def _run_episode(self):
        total_reward = total_bonus = total_success = time = 0
        state = self.env.reset()
        for time in range(self.horizon):
            self.exploration_policy.step_time()
            action = self.policy(state)
            next_state, reward, done, info = self.env.step(action)

            # bonus used only for logging, here
            bonus = 0.0
            if self.use_bonus:
                if info is not None and 'exploration_bonus' in info:
                    bonus = info['exploration_bonus']

            self.record(state, action, reward, next_state, done, info)
            state = next_state
            total_reward += reward
            total_bonus += bonus
            total_success += info.get("is_success", 0)
            if done:
                break
        return total_reward, total_bonus, total_success, time + 1

    def record(self, state, action, reward, next_state, done, info):
        """
        Record a transition by performing a Deep Q-Network iteration

        - push the transition into memory
        - sample a minibatch
        - compute the bellman residual loss over the minibatch
        - perform one gradient descent step
        - slowly track the policy network with the target network

        Parameters
        ----------
        state : object
        action : object
        reward : double
        next_state : object
        done : bool
        """
        if not self.training:
            return
        self.memory.push(state, action, reward, next_state, done, info)
        if self.memory.position % self.update_frequency == 0:
            self.update()

    def update(self):
        batch, weights, indexes = self.sample_minibatch()
        if batch:
            losses, target = self.compute_bellman_residual(batch)
            self.step_optimizer(losses.mean())
            if self.prioritized_replay:
                new_priorities = losses.abs().detach().cpu().numpy() + 1e-6
                self.memory.update_priorities(indexes, new_priorities)
            self.update_target_network()

    def policy(self, observation):
        """
        Act according to the state-action value model and an exploration
        policy

        Parameters

        :param observation: current obs
        :return: an action
        """
        values = self.get_state_action_values(observation)
        self.exploration_policy.update(values)
        return self.exploration_policy.sample()

    def sample_minibatch(self):
        if len(self.memory) < self.batch_size:
            return None, None, None
        if self.prioritized_replay:
            transitions, weights, indexes = self.memory.sample(self.batch_size)
        else:
            transitions, indexes = self.memory.sample(self.batch_size)
            weights = np.ones((self.batch_size,))
        return transitions, weights, indexes

    def update_target_network(self):
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())

    def step_optimizer(self, loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def compute_bellman_residual(self, batch):
        """
        Compute the Bellman Residuals over a batch

        Parameters
        ----------
        batch
            batch of transitions
        target_state_action_value
            if provided, acts as a target (s,a)-value
            if not, it will be computed from batch and model
            (Double DQN target)

        Returns
        -------
        The residuals over the batch, and the computed target.
        """
        # Concatenate the batch elements
        state = torch.cat(tuple(torch.tensor([batch.state],
                                             dtype=torch.float))).to(self.device)
        action = torch.tensor(batch.action,
                              dtype=torch.long).to(self.device)
        reward = torch.tensor(batch.reward,
                              dtype=torch.float).to(self.device)
        if self.use_bonus:
            bonus = self.env.bonus_batch(state, action).to(self.device) * self.exploration_policy.epsilon
            if self.writer:
                self.writer.add_scalar("debug/minibatch_mean_bonus", bonus.mean().item(), self.episode)
                self.writer.add_scalar("debug/minibatch_mean_reward", reward.mean().item(), self.episode)
            reward += bonus
        next_state = torch.cat(tuple(torch.tensor([batch.next_state],
                                                  dtype=torch.float))).to(self.device)
        terminal = torch.tensor(batch.terminal,
                                dtype=torch.bool).to(self.device)
        batch = Transition(state, action, reward, next_state, terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.value_net(batch.state)
        state_action_values = \
            state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)

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
                ).gather(1, best_actions.unsqueeze(1)) \
                    .squeeze(1)
            else:
                best_values, _ = self.target_net(batch.next_state).max(1)
            next_state_values[~batch.terminal] \
                = best_values[~batch.terminal]
            # Compute the expected Q values
            target_state_action_value = batch.reward + self.gamma * next_state_values

        # Compute residuals
        residuals = self.loss_function(state_action_values, target_state_action_value, reduction='none')
        return residuals, target_state_action_value

    def get_batch_state_values(self, states):
        """
        Get the state values of several states

        Parameters
        ----------
        states : array
            [s1; ...; sN] an array of states

        Returns
        -------
        values, actions:
            * [V1; ...; VN] the array of the state values for each state
            * [a1*; ...; aN*] the array of corresponding optimal action
            indexes for each state
        """
        values, actions = self.value_net(torch.tensor(states,
                                                      dtype=torch.float)
                                         .to(self.device)).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        """
        Get the state-action values of several states

        Parameters
        ----------
        states : array
            [s1; ...; sN] an array of states

        Returns
        -------
        values:[[Q11, ..., Q1n]; ...] the array of all action values
        for each state
        """
        return self.value_net(torch.tensor(states,
                                           dtype=torch.float)
                              .to(self.device)).data.cpu().numpy()

    def get_state_value(self, state):
        """
        Parameters
        ----------
        state : object
            s, an environment state
        Returns
        -------
        V, its state-value
        """
        values, actions = self.get_batch_state_values([state])
        return values[0], actions[0]

    def get_state_action_values(self, state):
        """
        Parameters
        ----------
        state : object
            s, an environment state

        Returns
        -------
            The array of its action-values for each actions.
        """
        return self.get_batch_state_action_values([state])[0]

    def reseed(self, seed_seq=None):
        """
        Get new random number generator for the agent.

        Parameters
        ----------
        seed_seq : np.random.SeedSequence, rlberry.seeding.Seeder or int, default : None
            Seed sequence from which to spawn the random number generator.
            If None, generate random seed.
            If int, use as entropy for SeedSequence.
            If seeder, use seeder.seed_seq
        """
        # self.seeder
        if seed_seq is None:
            self.seeder = self.seeder.spawn()
        else:
            self.seeder = Seeder(seed_seq)

        # Seed exploration policy
        self.exploration_policy.seed(self.seeder)

    def reset(self, **kwargs):
        self.episode = 0

    def action_distribution(self, state):
        values = self.get_state_action_values(state)
        self.exploration_policy.update(values)
        return self.exploration_policy.get_distribution()

    def set_time(self, time):
        self.exploration_policy.set_time(time)

    def eval_mode(self):
        self.training = False
        self.exploration_kwargs['method'] = "Greedy"
        self.exploration_policy = \
            exploration_factory(self.env.action_space,
                                **self.exploration_kwargs)

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
        self._writer = writer
        try:
            self.exploration_policy.set_writer(writer)
        except AttributeError:
            pass
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
