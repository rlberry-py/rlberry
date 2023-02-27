import torch
import torch.nn as nn

import gymnasium.spaces as spaces
import numpy as np
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.utils.replay import ReplayBuffer
from rlberry.agents.torch.utils.training import optimizer_factory
from rlberry.agents.torch.utils.models import default_policy_net_fn
from rlberry.agents.torch.utils.models import default_value_net_fn
from rlberry.utils.torch import choose_device
from rlberry.utils.factory import load
from typing import Optional

import rlberry

logger = rlberry.logger


class A2CAgent(AgentWithSimplePolicy):
    """
    Advantage Actor Critic Agent.

    A2C, or Advantage Actor Critic, is a synchronous version of the A3C policy
    gradient method. As an alternative to the asynchronous implementation of
    A3C, A2C is a synchronous, deterministic implementation that waits for each
    actor to finish its segment of experience before updating, averaging over
    all of the actors. This more effectively uses GPUs due to larger batch sizes.

    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
    batch_size : int
        Number of timesteps to wait before updating the policy.
    gamma : double
        Discount factor in [0, 1].
    entr_coef : double
        Entropy coefficient.
    learning_rate : double
        Learning rate.
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
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
    device : str
        Device to put the tensors on
    eval_interval : int, default = None
        Interval (in number of transitions) between agent evaluations in fit().
        If None, never evaluate.

    References
    ----------
    Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T., Harley, T.,
    Silver, D. & Kavukcuoglu, K. (2016).
    "Asynchronous methods for deep reinforcement learning."
    In International Conference on Machine Learning (pp. 1928-1937).
    """

    name = "A2C"

    def __init__(
        self,
        env,
        batch_size=256,
        gamma=0.99,
        entr_coef=0.01,
        learning_rate=0.01,
        optimizer_type="ADAM",
        policy_net_fn=None,
        value_net_fn=None,
        policy_net_kwargs=None,
        value_net_kwargs=None,
        device="cuda:best",
        eval_interval: Optional[int] = None,
        **kwargs
    ):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.batch_size = batch_size
        self.gamma = gamma
        self.entr_coef = entr_coef
        self.learning_rate = learning_rate
        self.device = choose_device(device)
        self.eval_interval = eval_interval

        self.policy_net_kwargs = policy_net_kwargs or {}
        self.value_net_kwargs = value_net_kwargs or {}

        if isinstance(policy_net_fn, str):
            self.policy_net_fn = load(policy_net_fn)
        elif policy_net_fn is None:
            self.policy_net_fn = default_policy_net_fn
        else:
            self.policy_net_fn = policy_net_fn

        if isinstance(value_net_fn, str):
            self.value_net_fn = load(value_net_fn)
        elif value_net_fn is None:
            self.value_net_fn = default_value_net_fn
        else:
            self.value_net_fn = value_net_fn

        self.optimizer_kwargs = {"optimizer_type": optimizer_type, "lr": learning_rate}

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)

        # get horizon
        if hasattr(self.env, "_max_episode_steps"):
            max_episode_steps = self.env._max_episode_steps
        else:
            max_episode_steps = np.inf
        self._max_episode_steps = max_episode_steps

        self._policy = None  # categorical policy function

        # initialize
        self.reset()

    def reset(self):
        self._policy = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(
            self.device
        )
        self._policy_optimizer = optimizer_factory(
            self._policy.parameters(), **self.optimizer_kwargs
        )

        self.value_net = self.value_net_fn(self.env, **self.value_net_kwargs).to(
            self.device
        )

        self.value_optimizer = optimizer_factory(
            self.value_net.parameters(), **self.optimizer_kwargs
        )

        self._policy_old = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(
            self.device
        )
        self._policy_old.load_state_dict(self._policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.memory = ReplayBuffer(max_replay_size=self.batch_size, rng=self.rng)
        self.memory.setup_entry("states", dtype=np.float32)
        if self._policy.ctns_actions:
            self.memory.setup_entry("actions", dtype=np.float32)
        else:
            self.memory.setup_entry("actions", dtype=int)
        self.memory.setup_entry("rewards", dtype=np.float32)
        self.memory.setup_entry("dones", dtype=bool)

        self.total_timesteps = 0
        self.total_episodes = 0

    def policy(self, observation):
        state = observation
        assert self._policy is not None
        state = torch.from_numpy(state).float().to(self.device)
        action_dist = self._policy_old(state)
        if self._policy.ctns_actions:
            action = action_dist.sample().numpy()
        else:
            action = action_dist.sample().item()
        return action

    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.

        Parameters
        ----------
        budget: int
            Number of timesteps to train the agent for.
            One step = one transition in the environment.
        """
        del kwargs
        timesteps_counter = 0
        episode_rewards = 0.0
        episode_timesteps = 0
        observation = self.env.reset()
        while timesteps_counter < budget:
            action = self._select_action(observation)
            next_obs, reward, done, _ = self.env.step(action)
            # if self._policy.ctns_actions:
            #     action = torch.from_numpy(action).float().to(self.device)
            # store data
            episode_rewards += reward
            self.memory.append(
                {
                    "states": observation,
                    "actions": action,
                    "rewards": reward,
                    "dones": done,
                }
            )

            # counters and next obs
            self.total_timesteps += 1
            timesteps_counter += 1
            episode_timesteps += 1
            observation = next_obs

            # update
            if self.total_timesteps % self.batch_size == 0:
                self._update()

            # eval
            total_timesteps = self.total_timesteps
            if (
                self.eval_interval is not None
                and total_timesteps % self.eval_interval == 0
            ):
                eval_rewards = self.eval(
                    eval_horizon=self._max_episode_steps, gamma=1.0
                )
                if self.writer:
                    memory_size = len(self.memory)
                    self.writer.add_scalar(
                        "eval_rewards", eval_rewards, total_timesteps
                    )
                    self.writer.add_scalar("memory_size", memory_size, total_timesteps)

            # check if episode ended
            if done:
                self.total_episodes += 1
                self.memory.end_episode()
                if self.writer:
                    self.writer.add_scalar(
                        "episode_rewards", episode_rewards, total_timesteps
                    )
                    self.writer.add_scalar(
                        "total_episodes", self.total_episodes, total_timesteps
                    )
                episode_rewards = 0.0
                episode_timesteps = 0
                observation = self.env.reset()

    def _select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action_dist = self._policy_old(state)
        action = action_dist.sample()
        if self._policy.ctns_actions:
            action = action.numpy()
        else:
            action = action.item()
        return action

    def _update(self):
        # monte carlo estimate of rewards
        rewards = []
        discounted_reward = 0

        memory_data = self.memory.data
        memory_states = memory_data["states"]
        memory_actions = memory_data["actions"]
        memory_rewards = memory_data["rewards"]
        memory_dones = memory_data["dones"]

        for reward, is_terminal in zip(
            reversed(memory_rewards), reversed(memory_dones)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # convert to tensor
        rewards = torch.FloatTensor(rewards).to(self.device)
        memory_states_tensors = [
            torch.tensor(states).to(self.device).float() for states in memory_states
        ]
        memory_actions_tensors = [
            torch.tensor(actions).to(self.device) for actions in memory_actions
        ]

        # convert list to tensor
        old_states = torch.stack(memory_states_tensors).to(self.device).detach()
        old_actions = torch.stack(memory_actions_tensors).to(self.device).detach()

        # evaluate old actions and values
        action_dist = self._policy(old_states)
        logprobs = action_dist.log_prob(old_actions)
        state_values = torch.squeeze(self.value_net(old_states))
        dist_entropy = action_dist.entropy()

        # normalize the advantages
        advantages = rewards - state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # find pg loss
        pg_loss = -logprobs * advantages
        loss = (
            pg_loss
            + 0.5 * self.MseLoss(state_values, rewards)
            - self.entr_coef * dist_entropy
        )

        # take gradient step
        self._policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        loss.mean().backward()

        self._policy_optimizer.step()
        self.value_optimizer.step()

        # copy new weights into old policy
        self._policy_old.load_state_dict(self._policy.state_dict())

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        batch_size = trial.suggest_categorical("batch_size", [1, 4, 8, 16, 32])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)

        entr_coef = trial.suggest_loguniform("entr_coef", 1e-8, 0.1)

        return {
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "entr_coef": entr_coef,
        }
