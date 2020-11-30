import numpy as np
import torch
import torch.nn as nn
import logging

import rlberry.seeding as seeding
import gym.spaces as spaces
from rlberry.agents import Agent
from rlberry.agents.utils.memories import CEMMemory
from rlberry.agents.utils.torch_training import optimizer_factory
from rlberry.agents.utils.torch_models import default_policy_net_fn
from rlberry.utils.writers import PeriodicWriter

logger = logging.getLogger(__name__)
# choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CEMAgent(Agent):
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
    batch_size : int
        Number of trajectories to sample at each iteration.
    percentile : int
        Percentile used to remove trajectories with low rewards.
    learning_rate : double
        Optimizer learning rate
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    policy_net_fn : function
        Function that returns an instance of a policy network (pytorch).
        If None, a default net is used.
    """

    name = "CrossEntropyAgent"
    fit_info = ("n_episodes", "episode_rewards")

    def __init__(self,
                 env,
                 n_episodes=1000,
                 horizon=100,
                 gamma=0.99,
                 batch_size=16,
                 percentile=70,
                 learning_rate=0.01,
                 optimizer_type='ADAM',
                 policy_net_fn=None,
                 **kwargs):
        Agent.__init__(self, env, **kwargs)

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.percentile = percentile
        self.learning_rate = learning_rate
        self.horizon = horizon

        # random number generator
        self.rng = seeding.get_rng()

        #
        self.policy_net_fn = policy_net_fn \
            or (lambda: default_policy_net_fn(self.env))

        self.optimizer_kwargs = {'optimizer_type': optimizer_type,
                                 'lr': learning_rate}

        # policy net
        self.policy_net = self.policy_net_fn().to(device)

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

    def fit(self, **kwargs):
        self.episode = 0
        info = {}
        self._rewards = np.zeros(self.n_episodes)
        self._cumul_rewards = np.zeros(self.n_episodes)

        for ep in range(self.n_episodes):
            episode_rewards = self._run_episode()
            self._rewards[ep] = episode_rewards
            self._cumul_rewards[ep] = episode_rewards + \
                self._cumul_rewards[max(0, ep - 1)]

            # update policy and clear batch
            loss, reward_mean, reward_bound = self._update()

        info["n_episodes"] = self.n_episodes
        info["episode_rewards"] = self._rewards
        return info

    def policy(self, observation, **kwargs):
        act_probs_v, scores = \
            self._get_action_probabilities_tensor(observation)
        act_probs = act_probs_v.data.numpy()[0]
        action = self.rng.choice(len(act_probs), p=act_probs)
        return action

    def _get_action_probabilities_tensor(self, observation):
        obs_v = torch.FloatTensor([observation]).to(device)
        sm = nn.Softmax(dim=1)
        scores = self.policy_net.action_scores(obs_v)
        act_probs_v = sm(scores)
        return act_probs_v, scores

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

        train_states_tensor = torch.FloatTensor(train_states).to(device)
        train_actions_tensor = torch.LongTensor(train_actions).to(device)

        # states in last trajectory
        last_states = self.memory.states[-1]
        last_states_tensor = torch.FloatTensor(last_states).to(device)

        return train_states_tensor, train_actions_tensor, \
            reward_bound, reward_mean, last_states_tensor

    def _run_episode(self):
        # interact for H steps
        episode_rewards = 0
        episode_states = []
        episode_actions = []
        state = self.env.reset()
        for hh in range(self.horizon):
            # take action according to policy_net
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)

            # save
            episode_states.append(state)
            episode_actions.append(action)

            # increment rewards
            episode_rewards += reward*np.power(self.gamma, hh)

            if done:
                break
            state = next_state

        self.memory.append(episode_states, episode_actions, episode_rewards)
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
        scores_last_traj = self.policy_net.action_scores(last_states_tensor)
        softmax = nn.Softmax(dim=1)
        # 1e-3 is added to avoid zeros
        probs_last_traj = softmax(scores_last_traj) + 1e-3
        entropy = - torch.sum((probs_last_traj * torch.log(probs_last_traj)),
                              dim=1)
        entropy = torch.mean(entropy)

        loss = loss - 0.1*entropy
        loss.backward()
        self.optimizer.step()

        return loss.item(), reward_mean, reward_bound
