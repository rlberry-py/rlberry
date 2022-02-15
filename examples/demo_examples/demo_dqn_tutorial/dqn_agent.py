"""
 =====================
 Demo: dqn_agent
 =====================
 Implements a simple DQN agent.
"""

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from rlberry.agents import Agent
from rlberry.utils.torch import choose_device


class ReplayBuffer:
    def __init__(self, capacity, rng):
        """
        Parameters
        ----------
        capacity : int
        Maximum number of transitions
        rng :
        instance of numpy's default_rng
        """
        self.capacity = capacity
        self.rng = rng  # random number generator
        self.memory = []
        self.position = 0

    def push(self, sample):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = self.rng.choice(len(self.memory), size=batch_size)
        samples = [self.memory[idx] for idx in indices]
        return map(np.asarray, zip(*samples))

    def __len__(self):
        return len(self.memory)


class DQNAgent(Agent):
    name = "DQN"

    def __init__(
        self,
        env,
        q_net_constructor,
        gamma: float = 0.99,
        batch_size: int = 256,
        eval_every: int = 500,
        buffer_capacity: int = 30000,
        update_target_every: int = 500,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        decrease_epsilon: int = 200,
        learning_rate: float = 0.001,
        device: str = "cuda:best",
        **kwargs
    ):
        Agent.__init__(self, env, **kwargs)
        env = self.env
        self.gamma = gamma
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.update_target_every = update_target_every
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decrease_epsilon = decrease_epsilon
        self.device_ = choose_device(device)
        self.total_timesteps = 0
        self.total_episodes = 0
        self.total_updates = 0

        # initialize epsilon
        self.epsilon = epsilon_start

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, self.rng)

        # create network and target network
        self.q_net = q_net_constructor(self.env)
        self.target_net = q_net_constructor(self.env)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # objective and optimizer
        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, evaluation=False):
        """
        If evaluation=False, get action according to exploration policy.
        Otherwise, get action according to the evaluation policy.
        """
        if self.rng.uniform() < self.epsilon and (not evaluation):
            return self.env.action_space.sample()
        else:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(self.device_)
            with torch.no_grad():
                Q = self.q_net(tensor_state)
                action_index = torch.argmax(Q, dim=1)
            return action_index.item()

    def fit(self, budget):
        """
        Parameters
        ----------
        budget : int
            Number of training timesteps
        """
        state = self.env.reset()
        done = False
        episode_reward = 0.0
        for _ in range(budget):
            self.total_timesteps += 1
            action = self.select_action(state, evaluation=False)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            self.replay_buffer.push((state, next_state, action, reward, done))

            if len(self.replay_buffer) > self.batch_size:
                #
                # Update model
                #
                self.total_updates += 1

                # get batch
                (
                    batch_state,
                    batch_next_state,
                    batch_action,
                    batch_reward,
                    batch_done,
                ) = self.replay_buffer.sample(self.batch_size)
                # convert to torch tensors
                batch_state = torch.FloatTensor(batch_state).to(self.device_)
                batch_next_state = torch.FloatTensor(batch_next_state).to(self.device_)
                batch_action = (
                    torch.LongTensor(batch_action).unsqueeze(1).to(self.device_)
                )
                batch_reward = (
                    torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device_)
                )
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device_)

                #
                # Compute loss and update nets
                #
                with torch.no_grad():
                    target_next = self.target_net(batch_next_state)
                    targets = (
                        batch_reward
                        + (1 - batch_done)
                        * self.gamma
                        * torch.max(target_next, dim=1, keepdim=True)[0]
                    )
                values = self.q_net(batch_state).gather(1, batch_action.long())
                loss = self.loss_fn(values, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar("loss", loss.item(), self.total_timesteps)

            # evaluate agent
            if self.total_timesteps % self.eval_every == 0:
                mean_rewards = self.eval(n_sim=5)
                self.writer.add_scalar(
                    "eval_rewards", mean_rewards, self.total_timesteps
                )

            # update target network
            if self.total_updates % self.update_target_every == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
                self.target_net.eval()

            # check end of episode
            state = next_state
            if done:
                state = self.env.reset()
                self.total_episodes += 1
                self.writer.add_scalar(
                    "episode_rewards", episode_reward, self.total_timesteps
                )
                self.writer.add_scalar(
                    "episode", self.total_episodes, self.total_timesteps
                )
                episode_reward = 0.0

                # decrease epsilon after end of episode
                if self.epsilon > self.epsilon_min:
                    self.epsilon -= (
                        self.epsilon_start - self.epsilon_min
                    ) / self.decrease_epsilon

    def eval(self, n_sim=1, **kwargs):
        del kwargs
        rewards = np.zeros(n_sim)
        eval_env = self.eval_env  # evaluation environment
        # Loop over number of simulations
        for sim in range(n_sim):
            state = eval_env.reset()
            done = False
            while not done:
                action = self.select_action(state, evaluation=True)
                next_state, reward, done, _ = eval_env.step(action)
                # update sum of rewards
                rewards[sim] += reward
                state = next_state
        return rewards.mean()
