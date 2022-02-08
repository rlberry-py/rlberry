import logging
from typing import Callable, Optional, Union

import gym
import numpy as np
import rlberry.agents.torch.td3.td3_utils as utils
import torch
from gym.spaces import Discrete
from rlberry import types
from rlberry.agents import Agent
from rlberry.agents.torch.td3.simple_replay_buffer import (
    SimpleReplayBuffer as ReplayBuffer,
)
from rlberry.agents.torch.td3.td3_env_wrapper import NormalizedContinuousEnvWrapper
from rlberry.utils.torch import choose_device


logger = logging.getLogger(__name__)


class TD3Agent(Agent):
    """Twin Delayed DDPG (TD3) agent.

    Paper: Fujimoto et al. (2018) https://arxiv.org/abs/1802.09477

    See also: https://spinningup.openai.com/en/latest/algorithms/td3.html

    Parameters
    ----------
    env : types.Env
        Environment.
    q_net_constructor : callable
        Constructor for Q-network. Takes an env as argument: q_net = q_net_constructor(env).
        Given a batch of observations of shape (batch_size, time_size, observation_dim),
        the Q-network's output must have the shape (batch_size, time_size, 2),
        where the last dimension represents the output of the two critic networks
        used by TD3.
    pi_net_constructor : callable
        Constructor for policy network. Takes an env as argument: pi_net = pi_net_constructor(env)
        Given a batch of observations of shape (batch_size, time_size, observation_dim),
        the policy network's output must have the shape (batch_size, time_size, action_dim),
        where action_dim is the dimension of the actions (if the action space is Box)
        or the number of actions (if the action space is Discrete).
    gamma : float
        Discount factor.
    batch_size : int
        Batch size (in number of chunks).
    chunk_size : int
        Size of trajectory chunks (i.e., subtrajectories) to sample from the replay buffer.
    target_update_parameter : int or float
        If int: interval (in number total number of online updates) between updates of the target network.
        If float: soft update coefficient
    learning_rate : float
        Optimizer learning rate.
    train_interval: int
        Update the model every ``train_interval`` steps.
        If -1, train only at the end of the episodes.
    gradient_steps: int
        How many gradient steps to do at each update.
        If -1, take the number of timesteps since last update.
    max_replay_size : int
        Maximum number of transitions in the replay buffer.
    learning_starts : int
        How many steps of the model to collect transitions for before learning starts
    eval_interval : int
        Interval (in number of transitions) between agent evaluations in fit().
        If None, never evaluate.
    lambda_ : float
        Q(lambda) parameter.
    policy_delay: int
        Policy networks will only be updated once every policy_delay steps
        per training steps.
        The Q values will be updated policy_delay more often (update every training step).
    action_noise: float
        Standard deviation of Gaussian noise added to behavior policy (for exploration).
    target_policy_noise: float
        Standard deviation of Gaussian noise added to target policy (smoothing noise).
    target_noise_clip: float
        Limit for absolute value of target policy smoothing noise.
    random_exploration_eps: float
        Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for TD3 normally but can help exploring.
    reg_coef: float
        Policy regularization coefficient.
    """

    name = "TD3"

    def __init__(
        self,
        env: types.Env,
        q_net_constructor: Callable[[gym.Env], torch.nn.Module],
        pi_net_constructor: Callable[[gym.Env], torch.nn.Module],
        gamma: float = 0.99,
        batch_size: int = 64,
        chunk_size: int = 8,
        target_update_parameter: Union[int, float] = 0.005,
        learning_rate: float = 1e-3,
        train_interval: int = 10,
        gradient_steps: int = -1,
        max_replay_size: int = 200_000,
        learning_starts: int = 10_000,
        eval_interval: Optional[int] = 500,
        lambda_: float = 0.5,
        policy_delay: int = 2,
        action_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        random_exploration_eps: float = 0.1,
        reg_coef: float = 1e-4,
        device: str = "cuda:best",
        **kwargs,
    ):
        Agent.__init__(self, env, **kwargs)

        # Preprocess env
        if isinstance(self.env.action_space, Discrete):
            self.action_process_fn = torch.nn.Softmax(dim=-1)
            logger.info(
                "[TD3] Wrapping environment with discrete actions: "
                "action_noise and target_policy_noise are now set to 0.0"
            )
            action_noise = 0.0
            target_policy_noise = 0.0
            self.action_is_prob = True
        else:
            self.action_is_prob = False
            self.action_process_fn = torch.tanh

        self.env = NormalizedContinuousEnvWrapper(self.env)
        self.eval_env = NormalizedContinuousEnvWrapper(self.eval_env)
        env = self.env

        # Torch device
        self.device = choose_device(device)

        # Parameters (general)
        self._gamma = gamma
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._target_update_parameter = target_update_parameter
        self._learning_rate = learning_rate
        self._max_replay_size = max_replay_size
        self._eval_interval = eval_interval
        self._learning_starts = learning_starts
        self._train_interval = train_interval
        self._gradient_steps = gradient_steps

        # Parameters (TD3)
        self.lambda_ = np.array(lambda_, dtype=np.float32)
        self.random_exploration_eps = random_exploration_eps
        self.policy_delay = policy_delay
        self.action_noise = action_noise
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.reg_coef = reg_coef

        # Initialize Q-Networks
        self.q_online = q_net_constructor(self.env).to(self.device)
        self.q_target = q_net_constructor(self.env).to(self.device)

        # Initialize pi-networks
        self.pi_online = pi_net_constructor(self.env).to(self.device)
        self.pi_target = pi_net_constructor(self.env).to(self.device)

        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            params=self.q_online.parameters(), lr=self._learning_rate
        )
        self.pi_optimizer = torch.optim.Adam(
            params=self.pi_online.parameters(), lr=self._learning_rate
        )

        # Objective for Q loss / KL regularization
        self.q_objective = torch.nn.MSELoss()

        # Checks
        assert self._target_update_parameter <= 1.0  # tau parameter for target updates

        #
        # Setup replay buffer
        #
        if hasattr(self.env, "_max_episode_steps"):
            max_episode_steps = self.env._max_episode_steps
        else:
            max_episode_steps = np.inf
        self._max_episode_steps = max_episode_steps

        self._replay_buffer = ReplayBuffer(
            max_replay_size, self.rng, self._max_episode_steps
        )
        self._replay_buffer.setup_entry("observations", np.float32)
        self._replay_buffer.setup_entry("actions", np.float32)
        self._replay_buffer.setup_entry("rewards", np.float32)
        self._replay_buffer.setup_entry("discounts", np.float32)
        self._replay_buffer.setup_entry("next_observations", np.float32)

        #
        # Counters
        #
        self._total_timesteps = 0
        self._total_episodes = 0
        self._total_updates = 0
        self._timesteps_since_last_update = 0

    def must_update(self, is_end_of_episode):
        """Returns true if the model must be updated in the current timestep,
        and the number of gradient steps to take"""
        total_timesteps = self._total_timesteps
        n_gradient_steps = self._gradient_steps

        if total_timesteps < self._learning_starts:
            return False, -1

        if n_gradient_steps == -1:
            n_gradient_steps = self._timesteps_since_last_update

        run_update = False
        if self._train_interval == -1:
            run_update = is_end_of_episode
        else:
            run_update = total_timesteps % self._train_interval == 0
        return run_update, n_gradient_steps

    def update(self, batch):
        """Update networks."""
        # Update counters
        self._total_updates += 1
        self._timesteps_since_last_update = 0

        # Batch
        observations = torch.tensor(batch["observations"]).to(
            self.device
        )  # (batch_size, chunk_size, ...)
        actions = torch.tensor(batch["actions"]).to(self.device)
        next_obs = torch.tensor(batch["next_observations"]).to(self.device)

        #
        # Critic update
        #

        # Compute target actions (tp1)
        noise = (
            torch.zeros_like(actions)
            .normal_(0, self.target_policy_noise)
            .to(self.device)
        )
        noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
        mu_target_tp1 = self.action_process_fn(self.pi_target(next_obs).detach())
        target_actions_tp1 = torch.clamp(mu_target_tp1 + noise, -1.0, 1.0)

        # Critic
        q_t = self.q_online(observations, actions)  # shape (batch, chunk, n_heads)
        v_tp1 = self.q_target(
            next_obs, target_actions_tp1
        ).detach()  # shape (batch, chunk, n_heads)

        # Compute returns
        v_tp1_min, _ = torch.min(v_tp1, dim=-1)
        lambda_returns = utils.lambda_returns(
            batch["rewards"],
            batch["discounts"],
            v_tp1_min.numpy().astype(np.float32),
            self.lambda_,
        )

        # Targets
        targets = torch.tensor(lambda_returns).to(self.device)

        # Update critics
        n_heads = q_t.shape[-1]
        q_loss = torch.tensor(0.0).to(self.device)
        for ii in range(n_heads):
            q_loss += self.q_objective(q_t[:, :, ii], targets)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        #
        # Delayed target and policy updates
        #
        if self._total_updates % self.policy_delay == 0:
            # Update policy
            actions_target_t = self.action_process_fn(
                self.pi_target(observations).detach()
            )
            actions_online_t = self.action_process_fn(self.pi_online(observations))
            q_vals = self.q_online(observations, actions_online_t)[
                :, :, 0
            ]  # taking the first head to fit the policy
            policy_loss = -q_vals.mean()
            # Add regularization with respect to target
            if self.action_is_prob:
                # compute KL
                eps = torch.tensor(1e-32).to(self.device)
                kl_wrt_target = (
                    actions_target_t
                    * (
                        torch.log(eps + actions_target_t)
                        - torch.log(eps + actions_online_t)
                    )
                ).sum(dim=-1)
            else:
                # compute MSE (KL between gaussians)
                kl_wrt_target = (
                    0.5
                    * torch.square(actions_online_t - actions_target_t).mean(dim=-1)
                    / (self.action_noise**2.0)
                )

            policy_reg_loss = kl_wrt_target.mean()
            policy_loss += self.reg_coef * policy_reg_loss

            self.pi_optimizer.zero_grad()
            policy_loss.backward()
            self.pi_optimizer.step()

            # target update
            tau = self._target_update_parameter
            for param, target_param in zip(
                self.q_online.parameters(), self.q_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )

            for param, target_param in zip(
                self.pi_online.parameters(), self.pi_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )

            # Write data
            if self.writer:
                self.writer.add_scalar("q_loss", q_loss.item(), self.total_timesteps)
                self.writer.add_scalar(
                    "policy_loss", policy_loss.item(), self.total_timesteps
                )
                self.writer.add_scalar(
                    "policy_reg_loss", policy_reg_loss.item(), self.total_timesteps
                )
                self.writer.add_scalar(
                    "total_updates", self._total_updates, self.total_timesteps
                )

    def policy(self, state, evaluation=False):
        """TO BE IMPLEMENTED BY CHILD CLASS
        state = embedding of [obs(0), action(0), ..., obs(t-1), action(t-1), obs(t)]
        """
        epsilon = self.random_exploration_eps
        if (not evaluation) and self.rng.uniform() < epsilon:
            return self.env.action_space.sample()
        else:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_mu = self.action_process_fn(self.pi_online(tensor_state))
                if not evaluation:
                    noise = (
                        torch.zeros_like(action_mu)
                        .normal_(0, self.action_noise)
                        .to(self.device)
                    )
                else:
                    noise = torch.zeros_like(action_mu).to(self.device)
                action = torch.clamp(action_mu + noise, -1.0, 1.0)[0, :]
            return action.numpy()

    @property
    def total_timesteps(self):
        return self._total_timesteps

    def fit(self, budget: int, **kwargs):
        del kwargs
        timesteps_counter = 0
        episode_rewards = 0.0
        episode_timesteps = 0
        observation = self.env.reset()
        while timesteps_counter < budget:
            if self.total_timesteps < self._learning_starts:
                action = self.env.action_space.sample()
            else:
                self._timesteps_since_last_update += 1
                action = self.policy(observation, evaluation=False)
            next_obs, reward, done, _ = self.env.step(action)

            # store data
            episode_rewards += reward
            self._replay_buffer.append(
                {
                    "observations": observation,
                    "actions": action,
                    "rewards": reward,
                    "discounts": self._gamma * (1.0 - done),
                    "next_observations": next_obs,
                }
            )

            # counters and next obs
            self._total_timesteps += 1
            timesteps_counter += 1
            episode_timesteps += 1
            observation = next_obs

            # update
            run_update, n_gradient_steps = self.must_update(done)
            if run_update:
                for _ in range(n_gradient_steps):
                    batch = self._replay_buffer.sample(
                        self._batch_size, self._chunk_size
                    )
                    if batch:
                        self.update(batch)

            # eval
            total_timesteps = self._total_timesteps
            if (
                self._eval_interval is not None
                and total_timesteps % self._eval_interval == 0
            ):
                eval_rewards = self.eval(
                    eval_horizon=self._max_episode_steps, gamma=1.0
                )
                if self.writer:
                    buffer_size = len(self._replay_buffer)
                    self.writer.add_scalar(
                        "eval_rewards", eval_rewards, total_timesteps
                    )
                    self.writer.add_scalar("buffer_size", buffer_size, total_timesteps)

            # check if episode ended
            if done:
                self._total_episodes += 1
                if self.writer:
                    self.writer.add_scalar(
                        "episode_rewards", episode_rewards, total_timesteps
                    )
                    self.writer.add_scalar(
                        "total_episodes", self._total_episodes, total_timesteps
                    )
                episode_rewards = 0.0
                episode_timesteps = 0
                observation = self.env.reset()

    def eval(self, eval_horizon=10**5, n_simimulations=5, gamma=1.0, **kwargs):
        del kwargs  # unused
        episode_rewards = np.zeros(n_simimulations)
        for sim in range(n_simimulations):
            observation = self.eval_env.reset()
            tt = 0
            while tt < eval_horizon:
                action = self.policy(observation, evaluation=True)
                next_obs, reward, done, _ = self.eval_env.step(action)
                episode_rewards[sim] += reward * np.power(gamma, tt)
                observation = next_obs
                tt += 1
                if done:
                    break
        return episode_rewards.mean()

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        gamma = trial.suggest_categorical("gamma", [0.95, 0.99, 0.995])
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        chunk_size = trial.suggest_categorical("chunk_size", [8, 16])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
        target_update_parameter = trial.suggest_categorical(
            "target_update_parameter", [0.001, 0.005]
        )
        train_interval = trial.suggest_categorical("train_interval", [-1, 1, 10, 100])
        eval_interval = trial.suggest_categorical("eval_interval", [None])
        lambda_ = trial.suggest_categorical("lambda_", [0.1, 0.5, 0.7])
        reg_coef = trial.suggest_loguniform("reg_coef", 1e-8, 1e-1)

        return dict(
            gamma=gamma,
            batch_size=batch_size,
            chunk_size=chunk_size,
            learning_rate=learning_rate,
            target_update_parameter=target_update_parameter,
            train_interval=train_interval,
            eval_interval=eval_interval,
            lambda_=lambda_,
            reg_coef=reg_coef,
        )
