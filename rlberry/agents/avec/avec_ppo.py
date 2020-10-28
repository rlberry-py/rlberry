import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import rlberry.spaces as spaces
from rlberry.agents import Agent
from rlberry.envs import OnlineModel


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        return action, dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


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


class AVECPPOAgent(Agent):
    """
    AVEC uses a modification of the training objective for the critic in actor-critic algorithms
    to better approximate the value function (critic). The new state-value function approximation
    learns the *relative* value of the states rather than their *absolute* value as in conventional
    actor-critic. This modification is:
    - well-motivated by recent studies [1,2];
    - theoretically sound;
    - intuitively supported by the need to improve the approximation error of the critic.

    The application of Actor with Variance Estimated Critic (AVEC) to state-of-the-art policy
    gradient methods produces considerable gains in performance (on average +26% for SAC and +40% for PPO)
    over the standard actor-critic training.

    References
    ----------
    Flet-Berliac, Y., Ouhamma, R., Maillard, O. A., & Preux, P. (2020).
    "Is Standard Deviation the New Standard? Revisiting the Critic in Deep Policy Gradients."
    arXiv preprint arXiv:2010.04440.

    [1] Ilyas, A., Engstrom, L., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L. & Madry, A. (2020).
    "A closer look at deep policy gradients."
    In International Conference on Learning Representations.

    [2] Tucker, G., Bhupatiraju, S., Gu, S., Turner, R., Ghahramani, Z. & Levine, S. (2018).
    "The mirage of action-dependent baselines in reinforcement learning."
    In International Conference on Machine Learning, pp. 5015–5024.
    """

    def __init__(self, env,
                 n_episodes=4000,
                 horizon=256,
                 gamma=0.99,
                 lr=0.0003,
                 eps_clip=0.2,
                 k_epochs=10,
                 verbose=1,
                 **kwargs):
        """
        env : OnlineModel
            Online model with continuous (Box) state space and discrete actions
        n_episodes : int
            Number of episodes
        horizon : int
            Horizon of the objective function. If None and gamma<1, set to 1/(1-gamma).
        gamma : double
            Discount factor in [0, 1]. If gamma is 1.0, the problem is set to be finite-horizon.
        lr : double
            Learning rate.
        eps_clip : double
            PPO clipping range (epsilon).
        k_epochs : int
            Number of epochs per update.
        verbose : int
            Controls the verbosity, if non zero, progress messages are printed.
        """
        Agent.__init__(self, env)
        self.id = "AVECPPO"
        self.fit_info = ("n_episodes", "episode_rewards")

        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.horizon = horizon
        self.n_episodes = n_episodes
        self.state_dim = self.env.observation_space.dim
        self.action_dim = self.env.action_space.n
        self.verbose = verbose

        # check environment
        assert isinstance(self.env, OnlineModel)
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        self.cat_policy = None  # categorical policy function

        # initialize
        self.reset()

    def reset(self, **kwargs):
        self.cat_policy = ActorCritic(self.state_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.cat_policy.parameters(), lr=self.lr, betas=(0.9, 0.999))

        self.cat_policy_old = ActorCritic(self.state_dim, self.action_dim).to(device)
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

        self.memory = Memory()

        self.episode = 0

        # logging config
        self._last_printed_ep = 0
        self._time_last_log = time.clock()
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

        return self._select_action(state)

    def fit(self, **kwargs):
        info = {}
        self._rewards = np.zeros(self.n_episodes)
        self._cumul_rewards = np.zeros(self.n_episodes)
        for k in range(self.n_episodes):
            episode_rewards = self._run_episode()
            self._rewards[k] = episode_rewards
            if k > 0:
                self._cumul_rewards[k] = episode_rewards + self._cumul_rewards[k - 1]
            self.episode += 1
            self._logging()

            # update
            self._update()
            self.memory.clear_memory()

        info["n_episodes"] = self.n_episodes
        info["episode_rewards"] = self._rewards
        return info

    def _logging(self):
        if self.verbose > 0:
            t_now = time.clock()
            time_elapsed = t_now - self._time_last_log
            if time_elapsed >= self._log_interval:
                self._time_last_log = t_now
                print(self._info_to_print())
                self._last_printed_ep = self.episode - 1

    def _info_to_print(self):
        prev_episode = self._last_printed_ep
        episode = self.episode - 1
        reward_per_ep = self._rewards[prev_episode:episode + 1].sum() / max(1, episode - prev_episode)
        time_per_ep = self._log_interval * 1000.0 / max(1, episode - prev_episode)
        fps = int((self.horizon / time_per_ep) * 1000)

        to_print = "[%s] episode = %d/%d | reward/ep = %0.2f | time/ep = %0.2f ms | fps = %i" % (
            self.id, episode, self.n_episodes, reward_per_ep, time_per_ep, fps)
        return to_print

    def _select_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        action, action_logprob = self.cat_policy_old.act(state)

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
            state, reward, done, _ = self.env.step(action)

            # save in batch
            self.memory.rewards.append(reward)
            self.memory.is_terminals.append(done)
            episode_rewards += reward

            if done:
                break

        return episode_rewards

    def _update(self):
        # monte carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
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
            logprobs, state_values, dist_entropy = self.cat_policy.evaluate(old_states, old_actions)

            # find ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # find surrogate loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self._avec_loss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # copy new weights into old policy
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

    def _avec_loss(self, y_pred, y_true):
        """
        Computes the objective function used in AVEC for the learning of the value function:
        the residual variance between the state-values and the empirical returns.

        Returns Var[y-ypred]
        :param y_pred: (np.ndarray) the prediction
        :param y_true: (np.ndarray) the expected value
        :return: (float) residual variance of ypred and y
        """
        assert y_true.ndim == 1 and y_pred.ndim == 1

        return torch.var(y_true - y_pred)
