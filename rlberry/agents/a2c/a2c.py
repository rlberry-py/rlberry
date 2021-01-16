import numpy as np
import torch
import torch.nn as nn
import logging

import gym.spaces as spaces
from rlberry.agents import IncrementalAgent
from rlberry.agents.utils.memories import Memory
from rlberry.agents.utils.torch_training import optimizer_factory
from rlberry.agents.utils.torch_models import default_policy_net_fn
from rlberry.agents.utils.torch_models import default_value_net_fn
from rlberry.utils.torch import choose_device
from rlberry.utils.writers import PeriodicWriter
from rlberry.wrappers.uncertainty_estimator_wrapper import UncertaintyEstimatorWrapper

logger = logging.getLogger(__name__)


class A2CAgent(IncrementalAgent):
    """
    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
    n_episodes : int
        Number of episodes
    batch_size : int
        Number of episodes to wait before updating the policy.
    horizon : int
        Horizon.
    gamma : double
        Discount factor in [0, 1].
    entr_coef : double
        Entropy coefficient.
    learning_rate : double
        Learning rate.
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    k_epochs : int
        Number of epochs per update.
    policy_net_fn : function(env, **kwargs)
        Function that returns an instance of a policy network (pytorch).
        If None, a default net is used.
    value_net_fn : function(env, **kwargs)
        Function that returns an instance of a value network (pytorch).
        If None, a default net is used.
    policy_net_kwargs : dict
        kwargs for policy_net_fn
    value_net_kwargs : dict
        kwargs for value_net_fn
    use_bonus : bool, default = False
        If true, compute an 'exploration_bonus' and add it to the reward.
        See also UncertaintyEstimatorWrapper.
    uncertainty_estimator_kwargs : dict
        Arguments for the UncertaintyEstimatorWrapper
    device : str
        Device to put the tensors on

    References
    ----------
    Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T., Harley, T.,
    Silver, D. & Kavukcuoglu, K. (2016).
    "Asynchronous methods for deep reinforcement learning."
    In International Conference on Machine Learning (pp. 1928-1937).
    """

    name = "A2C"

    def __init__(self, env,
                 n_episodes=4000,
                 batch_size=8,
                 horizon=256,
                 gamma=0.99,
                 entr_coef=0.01,
                 learning_rate=0.01,
                 optimizer_type='ADAM',
                 k_epochs=5,
                 policy_net_fn=None,
                 value_net_fn=None,
                 policy_net_kwargs=None,
                 value_net_kwargs=None,
                 use_bonus=False,
                 uncertainty_estimator_kwargs=None,
                 device="cuda:best",
                 **kwargs):
        self.use_bonus = use_bonus
        if self.use_bonus:
            env = UncertaintyEstimatorWrapper(env,
                                              **uncertainty_estimator_kwargs)
        IncrementalAgent.__init__(self, env, **kwargs)

        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.horizon = horizon
        self.gamma = gamma
        self.entr_coef = entr_coef
        self.learning_rate = learning_rate
        self.k_epochs = k_epochs
        self.device = choose_device(device)

        self.policy_net_kwargs = policy_net_kwargs or {}
        self.value_net_kwargs = value_net_kwargs or {}

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        #
        self.policy_net_fn = policy_net_fn or default_policy_net_fn
        self.value_net_fn = value_net_fn or default_value_net_fn

        self.optimizer_kwargs = {'optimizer_type': optimizer_type,
                                 'lr': learning_rate}

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        self.cat_policy = None  # categorical policy function

        # initialize
        self.reset()

    def reset(self, **kwargs):
        self.cat_policy = self.policy_net_fn(
                            self.env,
                            **self.policy_net_kwargs).to(self.device)
        self.policy_optimizer = optimizer_factory(
                                    self.cat_policy.parameters(),
                                    **self.optimizer_kwargs)

        self.value_net = self.value_net_fn(
                                    self.env,
                                    **self.value_net_kwargs).to(self.device)

        self.value_optimizer = optimizer_factory(
                                self.value_net.parameters(),
                                **self.optimizer_kwargs)

        self.cat_policy_old = self.policy_net_fn(
                                self.env,
                                **self.policy_net_kwargs).to(self.device)
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.memory = Memory()

        self.episode = 0

        # useful data
        self._rewards = np.zeros(self.n_episodes)
        self._cumul_rewards = np.zeros(self.n_episodes)

        # default writer
        log_every = 5*logger.getEffectiveLevel()
        self.writer = PeriodicWriter(self.name, log_every=log_every)

    def policy(self, state, **kwargs):
        assert self.cat_policy is not None
        state = torch.from_numpy(state).float().to(self.device)
        action_dist = self.cat_policy_old(state)
        action = action_dist.sample().item()
        return action

    def partial_fit(self, fraction: float, **kwargs):
        assert 0.0 < fraction <= 1.0
        n_episodes_to_run = int(np.ceil(fraction*self.n_episodes))
        count = 0
        while count < n_episodes_to_run and self.episode < self.n_episodes:
            self._run_episode()
            count += 1

        info = {"n_episodes": self.episode,
                "episode_rewards": self._rewards[:self.episode]}
        return info

    def _select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action_dist = self.cat_policy_old(state)
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)

        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob)

        return action.item()

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        for _ in range(self.horizon):
            # running policy_old
            action = self._select_action(state)
            next_state, reward, done, info = self.env.step(action)

            # check whether to use bonus
            bonus = 0.0
            if self.use_bonus:
                if info is not None and 'exploration_bonus' in info:
                    bonus = info['exploration_bonus']

            # save in batch
            self.memory.rewards.append(reward+bonus)   # add bonus here
            self.memory.is_terminals.append(done)
            episode_rewards += reward

            if done:
                break

            # update state
            state = next_state

        # update
        ep = self.episode
        self._rewards[ep] = episode_rewards
        self._cumul_rewards[ep] = episode_rewards \
            + self._cumul_rewards[max(0, ep - 1)]
        self.episode += 1
        #
        if self.writer is not None:
            self.writer.add_scalar("episode", self.episode, None)
            self.writer.add_scalar("ep reward", episode_rewards)

        #
        if self.episode % self.batch_size == 0:
            self._update()
            self.memory.clear_memory()

        return episode_rewards

    def _update(self):
        # monte carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards),
                                       reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # normalize the rewards
        rewards = torch.tensor(rewards).to(self.device).float()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()

        # optimize policy for K epochs
        for _ in range(self.k_epochs):
            # evaluate old actions and values
            action_dist = self.cat_policy(old_states)
            logprobs = action_dist.log_prob(old_actions)
            state_values = torch.squeeze(self.value_net(old_states))
            dist_entropy = action_dist.entropy()

            # normalize the advantages
            advantages = rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) \
                / (advantages.std() + 1e-8)
            # find pg loss
            pg_loss = - logprobs * advantages
            loss = pg_loss \
                + 0.5 * self.MseLoss(state_values, rewards) \
                - self.entr_coef * dist_entropy

            # take gradient step
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            loss.mean().backward()

            self.policy_optimizer.step()
            self.value_optimizer.step()

        # copy new weights into old policy
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        batch_size = trial.suggest_categorical('batch_size',
                                               [1, 4, 8, 16, 32])
        gamma = trial.suggest_categorical('gamma',
                                          [0.9, 0.95, 0.99])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)

        entr_coef = trial.suggest_loguniform('entr_coef', 1e-8, 0.1)

        k_epochs = trial.suggest_categorical('k_epochs',
                                             [1, 5, 10, 20])

        return {
                'batch_size': batch_size,
                'gamma': gamma,
                'learning_rate': learning_rate,
                'entr_coef': entr_coef,
                'k_epochs': k_epochs,
                }
