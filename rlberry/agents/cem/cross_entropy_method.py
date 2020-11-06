"""
Implements the cross-entropy method (CEM), as presented in the book "Deep Reinforcement Learning Hands-on" by Maxim Lapan.
The CE method aims to maximize the likelihood of actions that led to high rewards.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rlberry.spaces as spaces
from collections import namedtuple
from rlberry.agents import Agent


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class CEMAgent(Agent):
    def __init__(self, env, gamma=0.99, horizon=500, batch_size=16, n_batches=50, percentile=70, learning_rate=0.01, net=None, verbose=1, **kwargs):
        """
        Parameters
        ----------
        env : Model
            Environment for training.
        gamma : double
            Discount factor in [0, 1].
        horizon : int 
            Maximum length of a trajectory.
        batch_size : int
            Number of trajectories to sample at each iteration.
        n_batches : int 
            Number of total batches used for training.
        percentile : int 
            Percentile used to remove trajectories with low rewards.
        learning_rate : double 
            Optimizer learning rate 
        net : torch.nn.Module
            Policy net, input = state, output = vector such that (probability over actions) = softmax(output).
        verbose : int 
            Verbosity level.
        """
        Agent.__init__(self, env, **kwargs)
        self.name = 'CrossEntropyAgent'

        # check environment 
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.percentile = percentile
        self.learning_rate = learning_rate
        self.horizon = horizon  
        self.verbose = verbose 

        # policy
        self.net = net
        if net is None:  # default network
            hidden_size = 128
            obs_size = self.env.observation_space.high.shape[0]
            n_actions = self.env.action_space.n
            self.net = Net(obs_size, hidden_size, n_actions)

    def iterate_batches(self):
        batch = []
        episode_reward = 0.0
        episode_steps = []
        obs = self.env.reset()
        time = 0
        while True:
            action = self.policy(obs)
            next_obs, reward, is_done, _ = self.env.step(action)
            episode_reward += reward*np.power(self.gamma, time)
            episode_steps.append(EpisodeStep(observation=obs, action=action))
            if is_done or time > self.horizon:
                batch.append(Episode(reward=episode_reward, steps=episode_steps))
                episode_reward = 0.0
                episode_steps = []
                next_obs = self.env.reset()
                time = 0
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            obs = next_obs
            time += 1

    def filter_batch(self, batch):
        rewards = list(map(lambda s: s.reward, batch))
        reward_bound = np.percentile(rewards, self.percentile)
        reward_mean = float(np.mean(rewards))

        train_obs = []
        train_act = []
        for example in batch:
            if example.reward < reward_bound:
                continue
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))

        train_obs_v = torch.FloatTensor(train_obs)
        train_act_v = torch.LongTensor(train_act)
        return train_obs_v, train_act_v, reward_bound, reward_mean

    def policy(self, observation, **kwargs):
        obs_v = torch.FloatTensor([observation])
        sm = nn.Softmax(dim=1)
        act_probs_v = sm(self.net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        return action

    def fit(self, **kwargs):
        info = {}
        info["loss"] = np.zeros(self.n_batches)
        info["reward_bound"] = np.zeros(self.n_batches)
        info["reward_mean"]  = np.zeros(self.n_batches)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        objective = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.net.parameters(), lr=self.learning_rate)

        print("Training %s ..." % self.name)
        for iter_no, batch in enumerate(self.iterate_batches()):
            obs_v, acts_v, reward_b, reward_m = self.filter_batch(batch)
            optimizer.zero_grad()
            action_scores_v = self.net(obs_v)
            loss_v = objective(action_scores_v, acts_v)
            loss_v.backward()
            optimizer.step()
            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
                iter_no, loss_v.item(), reward_m, reward_b))

            if iter_no >= self.n_batches:
                print("...done.")
                return info
                
            info["loss"][iter_no] = loss_v.item()
            info["reward_bound"][iter_no] = reward_b
            info["reward_mean"][iter_no] = reward_m



if __name__ == "__main__":
    pass
