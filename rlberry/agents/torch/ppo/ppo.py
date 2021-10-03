import numpy as np
import torch
import torch.nn as nn
import logging

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.utils.memories import Memory
from rlberry.agents.torch.utils.training import optimizer_factory
from rlberry.agents.torch.utils.models import default_policy_net_fn
from rlberry.agents.torch.utils.models import default_value_net_fn
from rlberry.utils.torch import choose_device
from rlberry.wrappers.uncertainty_estimator_wrapper import UncertaintyEstimatorWrapper


logger = logging.getLogger(__name__)


class PPOAgent(AgentWithSimplePolicy):
    """
    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
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

    def __init__(self, env,
                 batch_size=64,
                 update_frequency=8,
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
                 **kwargs):  # TODO: sort arguments

        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        # bonus
        self.use_bonus = use_bonus
        if self.use_bonus:
            self.env = UncertaintyEstimatorWrapper(self.env, **uncertainty_estimator_kwargs)

        # algorithm parameters
        self.gamma = gamma
        self.horizon = horizon

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.k_epochs = k_epochs
        self.update_frequency = update_frequency

        self.eps_clip = eps_clip
        self.vf_coef = vf_coef
        self.entr_coef = entr_coef

        # options
        # TODO: add reward normalization option
        #       add observation normalization option
        #       add orthogonal weight initialization option
        #       add value function clip option
        #       add ... ?
        self.normalize_advantages = True  # TODO: turn into argument

        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        # function approximators
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
        self.cat_policy = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(self.device)
        self.policy_optimizer = optimizer_factory(self.cat_policy.parameters(), **self.optimizer_kwargs)

        self.value_net = self.value_net_fn(self.env, **self.value_net_kwargs).to(self.device)
        self.value_optimizer = optimizer_factory(self.value_net.parameters(), **self.optimizer_kwargs)

        self.cat_policy_old = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(self.device)
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

        self.MseLoss = nn.MSELoss()  # TODO: turn into argument

        self.memory = Memory()  # TODO: Improve memory to include returns and advantages
        self.returns = []  # TODO: add to memory
        self.advantages = []  # TODO: add to memory

        self.episode = 0

    def policy(self, observation):
        state = observation
        assert self.cat_policy is not None
        state = torch.from_numpy(state).float().to(self.device)
        action_dist = self.cat_policy_old(state)
        action = action_dist.sample().item()
        return action

    def fit(self, budget: int, **kwargs):
        del kwargs
        n_episodes_to_run = budget
        count = 0
        while count < n_episodes_to_run:
            self._run_episode()
            count += 1

    def _run_episode(self):
        # to store transitions
        states = []
        actions = []
        action_logprobs = []
        rewards = []
        is_terminals = []

        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()

        for _ in range(self.horizon):
            # running policy_old
            state = torch.from_numpy(state).float().to(self.device)

            action_dist = self.cat_policy_old(state)
            action = action_dist.sample()
            action_logprob = action_dist.log_prob(action)
            action = action

            next_state, reward, done, info = self.env.step(action.item())

            # check whether to use bonus
            bonus = 0.0
            if self.use_bonus:
                if info is not None and 'exploration_bonus' in info:
                    bonus = info['exploration_bonus']

            # save transition
            states.append(state)
            actions.append(action)
            action_logprobs.append(action_logprob)
            rewards.append(reward + bonus)  # bonus added here
            is_terminals.append(done)

            episode_rewards += reward

            if done:
                break

            # update state
            state = next_state

        # compute returns and advantages
        state_values = self.value_net(torch.stack(states).to(self.device)).detach()
        state_values = torch.squeeze(state_values).tolist()

        # TODO: add the option to normalize before computing returns/advantages?
        returns, advantages = self._compute_returns_avantages(rewards, is_terminals, state_values)

        # save in batch
        self.memory.states.extend(states)
        self.memory.actions.extend(actions)
        self.memory.logprobs.extend(action_logprobs)
        self.memory.rewards.extend(rewards)
        self.memory.is_terminals.extend(is_terminals)

        self.returns.extend(returns)  # TODO: add to memory (cf reset)
        self.advantages.extend(advantages)  # TODO: add to memory (cf reset)

        # increment ep counter
        self.episode += 1

        # log
        if self.writer is not None:
            self.writer.add_scalar("episode_rewards", episode_rewards, self.episode)

        # update
        if self.episode % self.update_frequency == 0:  # TODO: maybe change to update in function of n_steps instead
            self._update()
            self.memory.clear_memory()
            del self.returns[:]  # TODO: add to memory (cf reset)
            del self.advantages[:]  # TODO: add to memory (cf reset)

        return episode_rewards

    def _update(self):

        # convert list to tensor
        full_old_states = torch.stack(self.memory.states).to(self.device).detach()
        full_old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        full_old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()
        full_old_returns = torch.stack(self.returns).to(self.device).detach()
        full_old_advantages = torch.stack(self.advantages).to(self.device).detach()

        # optimize policy for K epochs
        n_samples = full_old_actions.size(0)
        n_batches = n_samples // self.batch_size

        for _ in range(self.k_epochs):

            # shuffle samples
            rd_indices = self.rng.choice(n_samples, size=n_samples, replace=False)
            shuffled_states = full_old_states[rd_indices]
            shuffled_actions = full_old_actions[rd_indices]
            shuffled_logprobs = full_old_logprobs[rd_indices]
            shuffled_returns = full_old_returns[rd_indices]
            shuffled_advantages = full_old_advantages[rd_indices]

            for k in range(n_batches):

                # sample batch
                batch_idx = np.arange(k * self.batch_size, min((k + 1) * self.batch_size, n_samples))
                old_states = shuffled_states[batch_idx]
                old_actions = shuffled_actions[batch_idx]
                old_logprobs = shuffled_logprobs[batch_idx]
                old_returns = shuffled_returns[batch_idx]
                old_advantages = shuffled_advantages[batch_idx]

                # evaluate old actions and values
                action_dist = self.cat_policy(old_states)
                logprobs = action_dist.log_prob(old_actions)
                state_values = torch.squeeze(self.value_net(old_states))
                dist_entropy = action_dist.entropy()

                # find ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs)

                # TODO: add this option
                # normalizing the rewards
                # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

                # normalize the advantages
                old_advantages = old_advantages.view(-1, )

                if self.normalize_advantages:
                    old_advantages = (old_advantages - old_advantages.mean()) / (old_advantages.std() + 1e-10)

                # compute surrogate loss
                surr1 = ratios * old_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * old_advantages
                surr_loss = torch.min(surr1, surr2)

                # compute value function loss
                loss_vf = self.vf_coef * self.MseLoss(state_values, old_returns)

                # compute entropy loss
                loss_entropy = self.entr_coef * dist_entropy

                # compute total loss
                loss = - surr_loss + loss_vf - loss_entropy

                # take gradient step
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                loss.mean().backward()

                self.policy_optimizer.step()
                self.value_optimizer.step()

        # log
        if self.writer:
            self.writer.add_scalar("fit/surrogate_loss", surr_loss.mean().cpu().detach().numpy(), self.episode)
            self.writer.add_scalar("fit/entropy_loss", dist_entropy.mean().cpu().detach().numpy(), self.episode)

        # copy new weights into old policy
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

    def _compute_returns_avantages(self, rewards, is_terminals, state_values):

        returns = torch.zeros(self.horizon).to(self.device)
        advantages = torch.zeros(self.horizon).to(self.device)

        if not self.use_gae:
            for t in reversed(range(self.horizon)):
                if t == self.horizon - 1:
                    returns[t] = rewards[t] + self.gamma * (1 - is_terminals[t]) * state_values[-1]
                else:
                    returns[t] = rewards[t] + self.gamma * (1 - is_terminals[t]) * returns[t + 1]

                advantages[t] = returns[t] - state_values[t]
        else:
            last_adv = 0
            for t in reversed(range(self.horizon)):
                if t == self.horizon - 1:
                    returns[t] = rewards[t] + self.gamma * (1 - is_terminals[t]) * state_values[-1]
                    td_error = returns[t] - state_values[t]
                else:
                    returns[t] = rewards[t] + self.gamma * (1 - is_terminals[t]) * returns[t + 1]
                    td_error = rewards[t] + self.gamma * (1 - is_terminals[t]) * state_values[t + 1] - state_values[t]

                last_adv = self.gae_lambda * self.gamma * (1 - is_terminals[t]) * last_adv + td_error
                advantages[t] = last_adv

        return returns, advantages

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
