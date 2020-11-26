import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import rlberry.spaces as spaces
from rlberry.agents import IncrementalAgent

# choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class AVECPPOAgent(IncrementalAgent):
    """
    AVEC uses a modification of the training objective for the critic in
    actor-critic algorithms to better approximate the value function (critic).
    The new state-value function approximation learns the *relative* value of
    the states rather than their *absolute* value as in conventional
    actor-critic. This modification is:
    - well-motivated by recent studies [1,2];
    - theoretically sound;
    - intuitively supported by the need to improve the approximation error
    of the critic.

    The application of Actor with Variance Estimated Critic (AVEC) to
    state-of-the-art policy gradient methods produces considerable
    gains in performance (on average +26% for SAC and +40% for PPO)
    over the standard actor-critic training.

    References
    ----------
    Flet-Berliac, Y., Ouhamma, R., Maillard, O. A., & Preux, P. (2020).
    "Is Standard Deviation the New Standard? Revisiting the Critic in Deep
    Policy Gradients."
    arXiv preprint arXiv:2010.04440.

    [1] Ilyas, A., Engstrom, L., Santurkar, S., Tsipras, D., Janoos, F.,
    Rudolph, L. & Madry, A. (2020).
    "A closer look at deep policy gradients."
    In International Conference on Learning Representations.

    [2] Tucker, G., Bhupatiraju, S., Gu, S., Turner, R., Ghahramani, Z. &
    Levine, S. (2018).
    "The mirage of action-dependent baselines in reinforcement learning."
    In International Conference on Machine Learning, pp. 5015–5024.
    """

    name = "AVECPPO"
    fit_info = ("n_episodes", "episode_rewards")

    def __init__(self, env,
                 n_episodes=4000,
                 batch_size=8,
                 horizon=256,
                 gamma=0.99,
                 entr_coef=0.01,
                 vf_coef=0.5,
                 learning_rate=0.0003,
                 eps_clip=0.2,
                 k_epochs=10,
                 verbose=1,
                 **kwargs):
        """
        env : Model
            model with continuous (Box) state space and discrete actions
        n_episodes : int
            Number of episodes
        batch_size : int
            Number of episodes to wait before updating the policy.
        horizon : int
            Horizon of the objective function. If None and gamma<1,
            set to 1/(1-gamma).
        gamma : double
            Discount factor in [0, 1]. If gamma is 1.0, the problem is set
            to be finite-horizon.
        entr_coef : double
            Entropy coefficient.
        vf_coef : double
            Value function loss coefficient.
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

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entr_coef = entr_coef
        self.vf_coef = vf_coef
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.horizon = horizon
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        
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
        self.cat_policy = ActorCritic(self.state_dim,
                                      self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.cat_policy.parameters(),
                                          lr=self.learning_rate, betas=(0.9, 0.999))

        self.cat_policy_old = ActorCritic(self.state_dim,
                                          self.action_dim).to(device)
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

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

        return self._select_action(state)

    def fit(self, **kwargs):
        for k in range(self.n_episodes):
            self._run_episode()

        info = {"n_episodes": self.episode, "episode_rewards": self._rewards[:self.episode]}
        return info

    def partial_fit(self, fraction, **kwargs):
        assert 0.0 < fraction <= 1.0
        n_episodes_to_run = int(np.ceil(fraction * self.n_episodes))
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
        reward_per_ep = self._rewards[prev_episode:episode + 1].sum() / \
                        max(1, episode - prev_episode)
        time_per_ep = self._log_interval * 1000.0 / \
                      max(1, episode - prev_episode)
        time_per_ep = max(0.01, time_per_ep)  # avoid div by zero
        fps = int((self.horizon / time_per_ep) * 1000)

        to_print = "[{}] episode = {}/{} ".format(self.name, episode + 1,
                                                  self.n_episodes) \
                   + "| reward/ep = {:0.2f} ".format(reward_per_ep) \
                   + "| time/ep = {:0.2f} ms".format(time_per_ep) \
                   + "| fps = {}".format(fps)
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
            logprobs, state_values, dist_entropy = \
                self.cat_policy.evaluate(old_states, old_actions)

            # find ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # normalize the advantages
            advantages = rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # find surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1
                                + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) \
                   + self.vf_coef * self._avec_loss(state_values, rewards) \
                   - self.entr_coef * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # copy new weights into old policy
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

    def _avec_loss(self, y_pred, y_true):
        """
        Computes the objective function used in AVEC for the learning
        of the value function:
        the residual variance between the state-values and the
        empirical returns.

        Returns Var[y-ypred]
        :param y_pred: (np.ndarray) the prediction
        :param y_true: (np.ndarray) the expected value
        :return: (float) residual variance of ypred and y
        """
        assert y_true.ndim == 1 and y_pred.ndim == 1

        return torch.var(y_true - y_pred)

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
