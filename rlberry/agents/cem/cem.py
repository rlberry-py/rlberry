import numpy as np
import torch
import torch.nn as nn
import logging

import gym.spaces as spaces
from rlberry.agents import IncrementalAgent
from rlberry.agents.utils.memories import CEMMemory
from rlberry.agents.utils.torch_training import optimizer_factory
from rlberry.agents.utils.torch_models import default_policy_net_fn
from rlberry.utils.torch import choose_device
from rlberry.utils.writers import PeriodicWriter

logger = logging.getLogger(__name__)


class CEMAgent(IncrementalAgent):
    """
    Parameters
    ----------
    env : Model
        Environment for training.
    n_episodes : int
        Number of training episodes.
    horizon : int
        Maximum length of a trajectory.
    gamma : double
        Discount factor in [0, 1].
    entr_coef : double
        Entropy coefficient.
    batch_size : int
        Number of trajectories to sample at each iteration.
    percentile : int
        Percentile used to remove trajectories with low rewards.
    learning_rate : double
        Optimizer learning rate
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    on_policy : bool
        If True, updates are done only with on-policy data.
    policy_net_fn : function(env, **kwargs)
        Function that returns an instance of a policy network (pytorch).
        If None, a default net is used.
    policy_net_kwargs : dict
        kwargs for policy_net_fn
    device : str
        Device to put the tensors on
    """

    name = "CrossEntropyAgent"

    def __init__(self,
                 env,
                 n_episodes=1000,
                 horizon=100,
                 gamma=0.99,
                 entr_coef=0.1,
                 batch_size=16,
                 percentile=70,
                 learning_rate=0.01,
                 optimizer_type='ADAM',
                 on_policy=False,
                 policy_net_fn=None,
                 policy_net_kwargs=None,
                 device="cuda:best",
                 **kwargs):
        IncrementalAgent.__init__(self, env, **kwargs)

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # parameters
        self.gamma = gamma
        self.entr_coef = entr_coef
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.percentile = percentile
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.on_policy = on_policy
        self.policy_net_kwargs = policy_net_kwargs or {}
        self.policy_net_fn = policy_net_fn or default_policy_net_fn
        self.optimizer_kwargs = {'optimizer_type': optimizer_type,
                                 'lr': learning_rate}
        self.device = choose_device(device)
        self.reset()

    def reset(self, **kwargs):
        # policy net
        self.policy_net = self.policy_net_fn(
                            self.env,
                            **self.policy_net_kwargs
                            ).to(self.device)

        # loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optimizer_factory(
                                    self.policy_net.parameters(),
                                    **self.optimizer_kwargs)

        # memory
        self.memory = CEMMemory(self.batch_size)

        # default writer
        self.writer = PeriodicWriter(self.name,
                                     log_every=5*logger.getEffectiveLevel())

        #
        self.episode = 0
        self._rewards = np.zeros(self.n_episodes)
        self._cumul_rewards = np.zeros(self.n_episodes)

    def partial_fit(self, fraction: float, **kwargs):
        assert 0.0 < fraction <= 1.0
        n_episodes_to_run = int(np.ceil(fraction * self.n_episodes))
        count = 0
        while count < n_episodes_to_run and self.episode < self.n_episodes:
            self._run_episode()
            count += 1

        info = {"n_episodes": self.episode,
                "episode_rewards": self._rewards[:self.episode]}
        return info

    def policy(self, observation, **kwargs):
        state = torch.from_numpy(observation).float().to(self.device)
        action_dist = self.policy_net(state)
        action = action_dist.sample().item()
        return action

    def _process_batch(self):
        rewards = np.array(self.memory.rewards)

        reward_bound = np.percentile(rewards, self.percentile)
        reward_mean = float(np.mean(rewards))

        train_states = []
        train_actions = []

        for ii in range(self.memory.size):
            if rewards[ii] < reward_bound:
                continue
            train_states.extend(self.memory.states[ii])
            train_actions.extend(self.memory.actions[ii])

        train_states_tensor = torch.FloatTensor(train_states).to(self.device)
        train_actions_tensor = torch.LongTensor(train_actions).to(self.device)

        # states in last trajectory
        last_states = self.memory.states[-1]
        last_states_tensor = torch.FloatTensor(last_states).to(self.device)

        return train_states_tensor, train_actions_tensor, \
            reward_bound, reward_mean, last_states_tensor

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        episode_states = []
        episode_actions = []
        state = self.env.reset()
        for _ in range(self.horizon):
            # take action according to policy_net
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)

            # save
            episode_states.append(state)
            episode_actions.append(action)

            # increment rewards
            episode_rewards += reward

            if done:
                break
            state = next_state

        self.memory.append(episode_states, episode_actions, episode_rewards)

        # update
        if (self.episode % self.batch_size == 0) or (not self.on_policy):
            self._update()

        #
        ep = self.episode
        self._rewards[ep] = episode_rewards
        self._cumul_rewards[ep] = episode_rewards + \
            self._cumul_rewards[max(0, ep - 1)]

        # increment ep and write
        self.episode += 1

        if self.writer is not None:
            self.writer.add_scalar("episode", self.episode, None)
            self.writer.add_scalar("ep reward", episode_rewards)

        return episode_rewards

    def _update(self):
        train_states_tensor, train_actions_tensor,\
            reward_bound, reward_mean, \
            last_states_tensor = self._process_batch()
        self.optimizer.zero_grad()
        action_scores = self.policy_net.action_scores(train_states_tensor)
        loss = self.loss_fn(action_scores, train_actions_tensor)

        # entropy in last trajectory
        action_dist_last_traj = self.policy_net(last_states_tensor)
        entropy = action_dist_last_traj.entropy().mean()

        loss = loss - self.entr_coef*entropy
        loss.backward()
        self.optimizer.step()

        return loss.item(), reward_mean, reward_bound

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        batch_size = trial.suggest_categorical('batch_size',
                                               [10, 20, 50, 100, 200])
        gamma = trial.suggest_categorical('gamma',
                                          [0.9, 0.95, 0.99])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)

        entr_coef = trial.suggest_loguniform('entr_coef', 1e-8, 0.1)

        on_policy = trial.suggest_categorical('on_policy',
                                              [False, True])
        return {
                'batch_size': batch_size,
                'gamma': gamma,
                'learning_rate': learning_rate,
                'entr_coef': entr_coef,
                'on_policy': on_policy
                }
