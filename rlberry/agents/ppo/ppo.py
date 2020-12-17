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


class PPOAgent(IncrementalAgent):
    """
    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
    n_episodes : int
        Number of episodes
    batch_size : int
        Number of *episodes* to wait before updating the policy.
    horizon : int
        Horizon.
    gamma : double
        Discount factor in [0, 1].
    entr_coef : double
        Entropy coefficient.
    vf_coef : double
        Value function loss coefficient.
    learning_rate : double
        Learning rate.
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    eps_clip : double
        PPO clipping range (epsilon).
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
    device: str
        Device to put the tensors on
    use_bonus : bool, default = False
        If true, compute the environment 'exploration_bonus'
        and add it to the reward. See also UncertaintyEstimatorWrapper.
    uncertainty_estimator_kwargs : dict
        kwargs for UncertaintyEstimatorWrapper

    References
    ----------
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017).
    "Proximal Policy Optimization Algorithms."
    arXiv preprint arXiv:1707.06347.

    Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015).
    "Trust region policy optimization."
    In International Conference on Machine Learning (pp. 1889-1897).
    """

    name = "PPO"
    fit_info = ("n_episodes", "episode_rewards")

    def __init__(self, env,
                 n_episodes=4000,
                 batch_size=8,
                 horizon=256,
                 gamma=0.99,
                 entr_coef=0.01,
                 vf_coef=0.5,
                 learning_rate=0.01,
                 optimizer_type='ADAM',
                 eps_clip=0.2,
                 k_epochs=5,
                 use_gae=True,
                 gae_lambda=0.95,
                 policy_net_fn=None,
                 value_net_fn=None,
                 policy_net_kwargs=None,
                 value_net_kwargs=None,
                 device="cuda:best",
                 use_bonus=False,
                 uncertainty_estimator_kwargs=None,
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
        self.vf_coef = vf_coef
        self.learning_rate = learning_rate
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.policy_net_kwargs = policy_net_kwargs or {}
        self.value_net_kwargs = value_net_kwargs or {}

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        #
        self.policy_net_fn = policy_net_fn or default_policy_net_fn
        self.value_net_fn = value_net_fn or default_value_net_fn

        self.device = choose_device(device)

        self.optimizer_kwargs = {'optimizer_type': optimizer_type,
                                 'lr': learning_rate}

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        self.cat_policy = None  # categorical policy function

        # initialize
        self.reset()

    @classmethod
    def from_config(cls, **kwargs):
        kwargs["policy_net_fn"] = eval(kwargs["policy_net_fn"])
        kwargs["value_net_fn"] = eval(kwargs["value_net_fn"])
        return cls(**kwargs)

    def reset(self, **kwargs):
        self.cat_policy = self.policy_net_fn(
                                self.env,
                                **self.policy_net_kwargs
                                ).to(self.device)
        self.policy_optimizer = optimizer_factory(
                                    self.cat_policy.parameters(),
                                    **self.optimizer_kwargs)

        self.value_net = self.value_net_fn(
                                self.env,
                                **self.value_net_kwargs
                                ).to(self.device)
        self.value_optimizer = optimizer_factory(
                                    self.value_net.parameters(),
                                    **self.optimizer_kwargs)

        self.cat_policy_old = self.policy_net_fn(
                                self.env,
                                **self.policy_net_kwargs
                                ).to(self.device)
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.memory = Memory()

        self.episode = 0

        # useful data
        self._rewards = np.zeros(self.n_episodes)
        self._cumul_rewards = np.zeros(self.n_episodes)

        # default writer
        self.writer = PeriodicWriter(self.name,
                                     log_every=5*logger.getEffectiveLevel())

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
            self.memory.rewards.append(reward+bonus)  # bonus added here
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
            self.writer.add_scalar("fit/total_reward", episode_rewards, self.episode)

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

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()

        # optimize policy for K epochs
        for _ in range(self.k_epochs):
            # evaluate old actions and values
            action_dist = self.cat_policy(old_states)
            logprobs = action_dist.log_prob(old_actions)
            state_values = torch.squeeze(self.value_net(old_states))
            dist_entropy = action_dist.entropy()

            # find ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            rewards = torch.tensor(rewards).to(self.device).float()
            returns = torch.zeros(rewards.shape).to(self.device)
            advantages = torch.zeros(rewards.shape).to(self.device)

            if not self.use_gae:
                for t in reversed(range(self.horizon)):
                    if t == self.horizon - 1:
                        returns[t] = rewards[t] + self.gamma * (1 - self.memory.is_terminals[t]) * state_values[-1]
                    else:
                        returns[t] = rewards[t] + self.gamma * (1 - self.memory.is_terminals[t]) * returns[t + 1]
                    advantages[t] = returns[t] - state_values[t]
            else:
                for t in reversed(range(self.horizon)):
                    if t == self.horizon - 1:
                        returns[t] = rewards[t] + self.gamma * (1 - self.memory.is_terminals[t]) * state_values[-1]
                        td_error = returns[t] - state_values[t]
                    else:
                        returns[t] = rewards[t] + self.gamma * (1 - self.memory.is_terminals[t]) * returns[t + 1]
                        td_error = rewards[t] + self.gamma * (1 - self.memory.is_terminals[t]) * state_values[t + 1] - state_values[t]
                    advantages[t] = advantages[t] * self.gae_lambda * self.gamma * (1 - self.memory.is_terminals[t]) + td_error

            # normalizing the rewards
            # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            # convert to pytorch tensors and move to gpu if available
            advantages = advantages.view(-1, )

            # normalize the advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            # find surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages
            surr_loss = torch.min(surr1, surr2)
            loss = - surr_loss \
                + self.vf_coef * self.MseLoss(state_values, rewards) \
                - self.entr_coef * dist_entropy

            # take gradient step
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            loss.mean().backward()

            self.policy_optimizer.step()
            self.value_optimizer.step()

        self.writer.add_scalar("fit/surrogate_loss", surr_loss.mean().cpu().detach().numpy(), self.episode)
        self.writer.add_scalar("fit/entropy_loss", dist_entropy.mean().cpu().detach().numpy(), self.episode)

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

        eps_clip = trial.suggest_categorical('eps_clip',
                                             [0.1, 0.2, 0.3])

        k_epochs = trial.suggest_categorical('k_epochs',
                                             [1, 5, 10, 20])

        return {
                'batch_size': batch_size,
                'gamma': gamma,
                'learning_rate': learning_rate,
                'entr_coef': entr_coef,
                'eps_clip': eps_clip,
                'k_epochs': k_epochs,
                }
