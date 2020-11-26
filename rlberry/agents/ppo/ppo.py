import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import rlberry.spaces as spaces
from rlberry.agents import IncrementalAgent

# choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(ValueNet, self).__init__()
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        state_value = self.critic(state)
        return torch.squeeze(state_value)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyNet, self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        action_probs = self.softmax(self.actor(state))
        dist = Categorical(action_probs)
        return dist

    def action_scores(self, state):
        action_scores = self.actor(state)
        return action_scores


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPOAgent(IncrementalAgent):
    """
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
                 learning_rate=0.01,
                 eps_clip=0.2,
                 k_epochs=5,
                 verbose=1,
                 **kwargs):
        """
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
        learning_rate : double
            Learning rate.
        eps_clip : double
            PPO clipping range (epsilon).
        k_epochs : int
            Number of epochs per update.
        verbose : int
            Controls the verbosity, if non zero, progress messages are printed.
        """
        IncrementalAgent.__init__(self, env, **kwargs)

        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.horizon = horizon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.state_dim = self.env.observation_space.dim
        self.action_dim = self.env.action_space.n
        self.verbose = verbose

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        self.cat_policy = None  # categorical policy function

        # initialize
        self.reset()

    def reset(self, **kwargs):
        self.cat_policy = PolicyNet(self.state_dim, self.action_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.cat_policy.parameters(),
                                                 lr=self.learning_rate,
                                                 betas=(0.9, 0.999))

        self.value_net = ValueNet(self.state_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(),
                                                lr=self.learning_rate,
                                                betas=(0.9, 0.999))

        self.cat_policy_old = \
            PolicyNet(self.state_dim, self.action_dim).to(device)
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.memory = Memory()

        self.episode = 0

        # useful data
        self._rewards = np.zeros(self.n_episodes)
        self._cumul_rewards = np.zeros(self.n_episodes)

        # logging config
        self._last_printed_ep = 0
        self._time_last_log = time.process_time()
        if self.verbose == 1:
            self._log_interval = 60  # in seconds
        elif self.verbose == 2:
            self._log_interval = 30
        elif self.verbose == 3:
            self._log_interval = 15
        elif self.verbose > 3:
            self._log_interval = 5

    def policy(self, state, **kwargs):
        assert self.cat_policy is not None
        state = torch.from_numpy(state).float().to(device)
        action_dist = self.cat_policy_old(state)
        action = action_dist.sample().item()
        return action

    def fit(self, **kwargs):
        for k in range(self.n_episodes):
            self._run_episode()

        info = {"n_episodes": self.episode, "episode_rewards": self._rewards[:self.episode]}
        return info

    def partial_fit(self, fraction, **kwargs):
        assert 0.0 < fraction <= 1.0
        n_episodes_to_run = int(np.ceil(fraction*self.n_episodes))
        count = 0
        while count < n_episodes_to_run and self.episode < self.n_episodes:
            self._run_episode()
            count += 1

        info = {"n_episodes": self.episode, "episode_rewards": self._rewards[:self.episode]}
        return info

    def _logging(self):
        if self.verbose > 0:
            t_now = time.process_time()
            time_elapsed = t_now - self._time_last_log
            if time_elapsed >= self._log_interval:
                self._time_last_log = t_now
                print(self._info_to_print())
                self._last_printed_ep = self.episode - 1

    def _info_to_print(self):
        prev_episode = self._last_printed_ep
        episode = self.episode - 1
        reward_per_ep = self._rewards[prev_episode:episode + 1].sum() \
            / max(1, episode - prev_episode)
        time_per_ep = self._log_interval * 1000.0 \
            / max(1, episode - prev_episode)
        time_per_ep = max(0.01, time_per_ep)  # avoid div by zero
        fps = int((self.horizon / time_per_ep) * 1000)

        to_print = "[{}] episode = {}/{} ".format(self.name, episode+1,
                                                  self.n_episodes) \
            + "| reward/ep = {:0.2f} ".format(reward_per_ep) \
            + "| time/ep = {:0.2f} ms".format(time_per_ep) \
            + "| fps = {}".format(fps)
        return to_print

    def _select_action(self, state):
        state = torch.from_numpy(state).float().to(device)
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
        for t in range(self.horizon):
            # running policy_old
            action = self._select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            # save in batch
            self.memory.rewards.append(reward)
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
        self._logging()

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

        # normalizing the rewards
        rewards = torch.tensor(rewards).to(device).float()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()

        # optimize policy for K epochs
        for _ in range(self.k_epochs):
            # evaluate old actions and values
            action_dist = self.cat_policy(old_states)
            logprobs = action_dist.log_prob(old_actions)
            state_values = self.value_net(old_states)
            dist_entropy = action_dist.entropy()

            # find ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # normalize the advantages
            advantages = rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # find surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) \
                + 0.5 * self.MseLoss(state_values, rewards) \
                - 0.01 * dist_entropy

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
                                               [1, 4, 8, 16, 32, 64])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
        return {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                }
