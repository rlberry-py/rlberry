import numpy as np
import torch
import torch.nn as nn

import gym.spaces as spaces
from rlberry.agents import IncrementalAgent
from rlberry.agents.utils.memories import Memory
from rlberry.agents.utils.torch_training import optimizer_factory
from rlberry.agents.utils.torch_models import default_policy_net_fn
from rlberry.agents.utils.torch_models import default_value_net_fn
from rlberry.utils.writers import PeriodicWriter

# choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class REINFORCEAgent(IncrementalAgent):
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
    learning_rate : double
        Learning rate.
    normalize: bool
        If True normalize rewards and advantages
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    policy_net_fn : function
        Function that returns an instance of a policy network (pytorch).
        If None, a default net is used.
    value_net_fn : function
        Function that returns an instance of a value network (pytorch).
        If None, a default net is used.
    verbose : int
        Controls the verbosity, if non zero, progress messages are printed.


    References
    ----------
    Williams, Ronald J.,
    "Simple statistical gradient-following algorithms for connectionist
    reinforcement learning."
    ReinforcementLearning.Springer,Boston,MA,1992.5-3
    """

    name = "REINFORCE"
    fit_info = ("n_episodes", "episode_rewards")

    def __init__(self, env,
                 n_episodes=4000,
                 batch_size=8,
                 horizon=256,
                 gamma=0.99,
                 learning_rate=0.0001,
                 normalize=False,
                 optimizer_type='ADAM',
                 policy_net_fn=None,
                 value_net_fn=None,
                 verbose=5,
                 **kwargs):
        IncrementalAgent.__init__(self, env, **kwargs)

        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.horizon = horizon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.normalize = normalize

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.verbose = verbose

        #
        self.policy_net_fn = policy_net_fn \
            or (lambda: default_policy_net_fn(self.env))

        self.value_net_fn = value_net_fn \
            or (lambda: default_value_net_fn(self.env))

        self.optimizer_kwargs = {'optimizer_type': optimizer_type,
                                 'lr': learning_rate}

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        self.policy_net = None  # policy network

        # initialize
        self.reset()

    def reset(self, **kwargs):
        self.policy_net = self.policy_net_fn().to(device)
        self.policy_optimizer = optimizer_factory(
                                    self.policy_net.parameters(),
                                    **self.optimizer_kwargs)

        self.value_net = self.value_net_fn().to(device)
        self.value_optimizer = optimizer_factory(
                                    self.value_net.parameters(),
                                    **self.optimizer_kwargs)

        self.MseLoss = nn.MSELoss()

        self.memory = Memory()

        self.episode = 0

        # useful data
        self._rewards = np.zeros(self.n_episodes)
        self._cumul_rewards = np.zeros(self.n_episodes)

        # default writer
        log_every = 0
        if self.verbose > 0:
            log_every = 200/self.verbose
        self.writer = PeriodicWriter(self.name, log_every=log_every)

    def policy(self, state, **kwargs):
        assert self.policy_net is not None
        state = torch.from_numpy(state).float().to(device)
        action_dist = self.policy_net(state)
        action = action_dist.sample().item()
        return action

    def fit(self, **kwargs):
        for _ in range(self.n_episodes):
            self._run_episode()

        info = {"n_episodes": self.episode,
                "episode_rewards": self._rewards[:self.episode]}
        return info

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

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        for _ in range(self.horizon):
            # running policy
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)

            # save in batch
            self.memory.states.append(state)
            self.memory.actions.append(action)
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

        #
        if self.writer is not None:
            self.writer.add_scalar("episode", self.episode, None)
            self.writer.add_scalar("ep reward", episode_rewards)

        #
        if self.episode % self.batch_size == 0:
            self._update()
            self.memory.clear_memory()

        return episode_rewards

    def _normalize(self, x):
        return (x-x.mean())/(x.std()+1e-5)

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
        states = torch.FloatTensor(self.memory.states).to(device)
        actions = torch.LongTensor(self.memory.actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        if self.normalize:
            rewards = self._normalize(rewards)
        # evaluate logprobs and values
        action_dist = self.policy_net(states)
        logprobs = action_dist.log_prob(actions)
        state_values = self.value_net(states)

        # compute advantages
        advantages = rewards - state_values.detach()
        if self.normalize:
            advantages = self._normalize(advantages)
        # compute loss
        loss = -logprobs * advantages + self.MseLoss(state_values, rewards)

        # take gradient step
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        loss.mean().backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

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
