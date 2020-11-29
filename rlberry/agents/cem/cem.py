import time

import numpy as np
import torch
import torch.nn as nn

import rlberry.seeding as seeding
import gym.spaces as spaces
from rlberry.agents import Agent
from rlberry.agents.utils.memories import CEMMemory

# choose device
from rlberry.agents.utils.torch_models import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CEMAgent(Agent):

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
                 verbose=5,
                 **kwargs):
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
        verbose : int
            Verbosity level.
        """
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
        self.verbose = verbose

        # random number generator
        self.rng = seeding.get_rng()

        # policy net
        hidden_size = 128
        obs_size = self.env.observation_space.high.shape[0]
        n_actions = self.env.action_space.n
        self.policy_net = Net(obs_size, hidden_size, n_actions).to(device)

        # loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.policy_net.parameters(),
                                          lr=self.learning_rate)

        # memory
        self.memory = CEMMemory(self.batch_size)

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
            self.episode += 1
            self._logging()

        info["n_episodes"] = self.n_episodes
        info["episode_rewards"] = self._rewards
        return info

    def _logging(self):
        if self.verbose > 0:
            t_now = time.process_time()
            time_elapsed = t_now - self._time_last_log
            if (time_elapsed >= self._log_interval) \
                    or (self.episode == self.n_episodes):
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

    def policy(self, observation, **kwargs):
        act_probs_v, scores = \
            self._get_action_probabilities_tensor(observation)
        act_probs = act_probs_v.data.numpy()[0]
        action = self.rng.choice(len(act_probs), p=act_probs)
        return action

    def _get_action_probabilities_tensor(self, observation):
        obs_v = torch.FloatTensor([observation]).to(device)
        sm = nn.Softmax(dim=1)
        scores = self.policy_net(obs_v)
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

        return episode_rewards

    def _update(self):
        train_states_tensor, train_actions_tensor,\
            reward_bound, reward_mean, \
            last_states_tensor = self._process_batch()
        self.optimizer.zero_grad()
        action_scores = self.policy_net(train_states_tensor)
        loss = self.loss_fn(action_scores, train_actions_tensor)

        # entropy in last trajectory
        scores_last_traj = self.policy_net(last_states_tensor)
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
