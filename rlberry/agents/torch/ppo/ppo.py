import numpy as np
import torch
import torch.nn as nn

import gymnasium.spaces as spaces
import rlberry
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents import AgentTorch
from rlberry.envs.utils import process_env
from rlberry.agents.torch.utils.training import optimizer_factory
from rlberry.agents.torch.utils.models import default_policy_net_fn
from rlberry.agents.torch.utils.models import default_value_net_fn
from rlberry.utils.torch import choose_device
from rlberry.utils.factory import load
from rlberry.agents.torch.ppo.ppo_utils import (
    process_ppo_env,
    lambda_returns,
    RolloutBuffer,
)

import dill
import pickle
import bz2
import _pickle as cPickle
from pathlib import Path


logger = rlberry.logger


# Notes about VecEnvs:
# - reset() returns a numpy array of shape (n_envs, state_dim)
# - step() returns a tuple of arrays (states, rewards, dones, infos)
#   - states: np.array (n_envs, state_dim) dtype varies
#   - rewards: np.array (n_envs,) np.float64
#   - dones: np.array (n_envs,) bool
#   - infos: list (n_envs,) dict
# - close() closes all environments


class PPOAgent(AgentTorch, AgentWithSimplePolicy):
    """
    Proximal Policy Optimization Agent.

    Policy gradient methods for reinforcement learning, which alternate between
    sampling data through interaction with the environment, and optimizing a
    “surrogate” objective function using stochastic gradient ascent.

    Parameters
    ----------
    env : rlberry Env
        Environment with continuous (Box) observation space.
    n_envs: int
        Number of environments to be used.
    n_steps : int
        Number of transitions to collect in each environment per update.
    batch_size : int
        Size of mini batches during each PPO update epoch. It is recommended
        that n_envs * n_steps is divisible by batch_size.
    gamma : float
        Discount factor in [0, 1].
    k_epochs : int
        Number of PPO epochs per update.
    clip_eps : float
        PPO clipping range (epsilon).
    target_kl: float
        Target KL divergence. If KL divergence between the current policy and
        the new policy is greater than target_kl, the update is stopped early.
        Set to None to disable early stopping.
    normalize_avantages : bool
        Whether or not to normalize advantages.
    gae_lambda : float
        Lambda parameter for TD(lambda) and Generalized Advantage Estimation.
    entr_coef : float
        Entropy coefficient.
    vf_coef : float
        Value function loss coefficient.
    value_loss: str
        Type of value loss. 'mse' corresponds to mean squared error,
        'clipped' corresponds to the original PPO loss, and 'avec'
        corresponds to the AVEC loss (Flet-Berliac et al. 2021).
    max_grad_norm : float
        Maximum norm of the gradient of both actor and critic networks.
    learning_rate : float
        Learning rate.
    lr_schedule: str
        Learning rate schedule. 'constant' corresponds to a constant learning
        rate, and 'linear' corresponds to a linearly decreasing learning rate,
        starting at learning_rate and ending at 0. WARNING: the schedule is
        reset at each call to fit().
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    policy_net_fn : function(env, **kwargs)
        Function that returns an instance of a policy network (pytorch).
        If None, a default net is used.
    policy_net_kwargs : dict
        kwargs for policy_net_fn
    value_net_fn : function(env, **kwargs)
        Function that returns an instance of a value network (pytorch).
        If None, a default net is used.
    value_net_kwargs : dict
        kwargs for value_net_fn
    eval_env : rlberry Env
        Environment used for evaluation. If None, env is used.
    n_eval_episodes : int
        Number of episodes to be used for evaluation.
    eval_horizon : int
        Maximum number of steps per episode during evaluation.
    eval_freq : int
        Number of updates between evaluations. If None, no evaluation is
        performed.
    device: str
        Device on which to put the tensors. 'cuda:best' by default.
    **kwargs : Keyword Arguments
         Arguments to be passed to `AgentWithSimplePolicy.__init__(self, env, **kwargs)` (:class:`~rlberry.agents.AgentWithSimplePolicy`).

    Attributes
    ----------
    __value_losses__ : list
        List of supported value loss types. ["clipped", "mse", "avec"]
    __lr_schedule___ : list
        List of supported learning rate schedule types. ["constant", "linear"]
    copy_env : bool
        If True, copy the environment to create multiple environments for parallel interaction.
    n_envs : int
        Number of environments used by the agent.
    n_steps : int
        Number of transitions to collect in each environment per update.
    batch_size : int
        Size of mini batches during each PPO update epoch.
    gamma : float
        Discount factor used to discount future rewards.
    k_epochs : int
        Number of PPO epochs per update.
    clip_eps : float
        PPO clipping range (epsilon).
    target_kl: float
        Target KL divergence for early stopping. If None, early stopping is disabled.
    normalize_advantages : bool
        Whether or not to normalize advantages.
    gae_lambda : float
        Lambda parameter for TD(lambda) and Generalized Advantage Estimation.
    entr_coef : float
        Entropy coefficient. Controls the contribution of entropy regularization to the policy's objective.
    vf_coef : float
        Value function loss coefficient. Controls the contribution of the value function loss to the total loss.
    value_loss: str
        Type of value loss used. Can be "mse", "clipped", or "avec".
    max_grad_norm : float
        Maximum norm of the gradient of both actor and critic networks. Used for gradient clipping.
    learning_rate : float
        Learning rate used by the optimizer during neural network training.
    lr_schedule : str
        Learning rate schedule used during training. Can be "constant" or "linear".
    optimizer_type : str
        Type of optimizer used during neural network training.
    n_eval_episodes : int
        Number of episodes used for evaluation.
    eval_horizon : int
        Maximum number of steps per episode during evaluation.
    eval_freq : int
        Number of updates between evaluations. If None, no evaluation is performed.
    policy_net_fn : function(env, **kwargs)
        Function that returns an instance of a policy network (PyTorch).
    policy_net_kwargs : dict
        Keyword arguments for `policy_net_fn`.
    value_net_fn : function(env, **kwargs)
        Function that returns an instance of a value network (PyTorch).
    value_net_kwargs : dict
        Keyword arguments for `value_net_fn`.
    eval_env : rlberry.Env
        The environment used for evaluation. If None, the same environment as env is used.
    state_dim : int
        Dimensionality of the continuous state space of the environment.
    policy_net : torch.nn.Module
        The policy network used by the agent.
    value_net : torch.nn.Module
        The value network used by the agent.
    device : str
        Torch device on which the agent's neural networks are placed.
    optimizer_kwargs : dict
        Keyword arguments for the optimizer used during neural network training.

    References
    ----------
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017).
    "Proximal Policy Optimization Algorithms."
    arXiv preprint arXiv:1707.06347.

    Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015).
    "Trust region policy optimization."
    In International Conference on Machine Learning (pp. 1889-1897).

    Flet-Berliac, Y., Ouhamma, R., Maillard, O.-A., Preux, P. (2021)
    "Learning Value Functions in Deep Policy Gradients using Residual Variance."
    In 9th International Conference on Learning Representations (ICLR).
    """

    name = "PPO"
    __value_losses__ = ["clipped", "mse", "avec"]
    __lr_schedule___ = ["constant", "linear"]

    def __init__(
        self,
        env,
        copy_env=True,
        n_envs=1,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        k_epochs=10,
        clip_eps=0.2,
        target_kl=0.05,
        normalize_advantages=True,
        gae_lambda=0.95,
        entr_coef=0.01,
        vf_coef=0.5,
        value_loss="mse",
        max_grad_norm=0.5,
        learning_rate=3e-4,
        lr_schedule="constant",
        optimizer_type="ADAM",
        policy_net_fn=None,
        policy_net_kwargs=None,
        value_net_fn=None,
        value_net_kwargs=None,
        eval_env=None,
        n_eval_episodes=10,
        eval_horizon=int(1e5),
        eval_freq=None,
        device="cuda:best",
        **kwargs
    ):
        kwargs.pop("eval_env", None)
        AgentWithSimplePolicy.__init__(
            self, None, **kwargs
        )  # PPO handles the env internally

        # create environment
        self.copy_env = copy_env
        self.n_envs = n_envs
        self.env = process_ppo_env(env, self.seeder, num_envs=n_envs, copy_env=copy_env)
        eval_env = eval_env or env
        self.eval_env = process_env(eval_env, self.seeder, copy_env=copy_env)

        # hyperparameters
        value_loss, lr_schedule = value_loss.lower(), lr_schedule.lower()
        assert value_loss in self.__value_losses__, "value_loss must be in {}".format(
            self.__value_losses__
        )
        assert lr_schedule in self.__lr_schedule___, "lr_schedule must be in {}".format(
            self.__lr_schedule___
        )

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.clip_eps = clip_eps
        self.target_kl = target_kl
        self.normalize_advantages = normalize_advantages
        self.gae_lambda = gae_lambda
        self.entr_coef = entr_coef
        self.vf_coef = vf_coef
        self.value_loss = value_loss
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.optimizer_type = optimizer_type
        self.n_eval_episodes = n_eval_episodes
        self.eval_horizon = eval_horizon
        self.eval_freq = eval_freq
        self.kwargs = kwargs

        self.state_dim = self.env.observation_space.shape[0]

        # policy network
        self.policy_net_kwargs = policy_net_kwargs or {}
        if isinstance(policy_net_fn, str):
            self.policy_net_fn = load(policy_net_fn)
        elif policy_net_fn is None:
            self.policy_net_fn = default_policy_net_fn
        else:
            self.policy_net_fn = policy_net_fn

        # value network
        self.value_net_kwargs = value_net_kwargs or {}
        if isinstance(value_net_fn, str):
            self.value_net_fn = load(value_net_fn)
        elif value_net_fn is None:
            self.value_net_fn = default_value_net_fn
        else:
            self.value_net_fn = value_net_fn

        self.device = choose_device(device)

        self.optimizer_kwargs = {
            "optimizer_type": optimizer_type,
            "lr": learning_rate,
            "eps": 1e-5,
        }

        # check environment
        # TODO: should we restrict this to Box?
        #       what about the action space?
        assert isinstance(self.env.observation_space, spaces.Box)

        # initialize
        self.policy_net = self.value_net = None
        self.reset()

    @classmethod
    def from_config(cls, **kwargs):
        kwargs["policy_net_fn"] = eval(kwargs["policy_net_fn"])
        kwargs["value_net_fn"] = eval(kwargs["value_net_fn"])
        return cls(**kwargs)

    def reset(self, **kwargs):
        """
        Reset the agent.
        """
        self.total_timesteps = 0
        self.total_episodes = 0

        # Initialize rollout buffer
        self.memory = RolloutBuffer(self.rng, self.n_steps)
        self.memory.setup_entry("observations", dtype=np.float32)
        self.memory.setup_entry("actions", dtype=self.env.single_action_space.dtype)
        self.memory.setup_entry("rewards", dtype=np.float32)
        self.memory.setup_entry("dones", dtype=bool)
        self.memory.setup_entry("logprobs", dtype=np.float32)
        self.memory.setup_entry("infos", dtype=dict)

        # Initialize neural networks and optimizers
        # TODO: using a single env to configure the networks is a hack that
        #       should be fixed when model factories are revised
        env = self.env.envs[0]
        self.policy_net = self.policy_net_fn(env, **self.policy_net_kwargs).to(
            self.device
        )
        self.value_net = self.value_net_fn(env, **self.value_net_kwargs).to(self.device)
        self.optimizer = optimizer_factory(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            **self.optimizer_kwargs
        )

    def policy(self, observation):
        assert self.policy_net is not None
        obs = torch.from_numpy(observation).float().to(self.device)
        action = self.policy_net(obs).sample()
        return action.cpu().numpy()

    def fit(self, budget: int, lr_scheduler=None, **kwargs):
        """
        Train the agent using the provided environment.

        Parameters
        ----------
        budget: int
            Total number of steps to be performed in the environment. Parameters
            are updated every n_steps interactions with the environment.
        lr_scheduler: callable
            A function that takes the current step and returns the current learning
            rate. If None, a default scheduler is used.
        **kwargs : Keyword Arguments
            Extra arguments. Not used for this agent.
        """
        del kwargs

        if lr_scheduler is None:
            lr_scheduler = self._get_lr_scheduler(budget)

        if len(self.memory) == 0:
            timesteps_counter = 0
        else:  # it's not the first "fit" on this agent, so there is a previous buffer to continue
            timesteps_counter = len(self.memory) * self.n_envs

        episode_returns = np.zeros(self.n_envs, dtype=np.float32)
        episode_lengths = np.zeros(self.n_envs, dtype=np.int32)

        next_obs, infos = self.env.reset()
        next_obs = torch.Tensor(next_obs).to(
            self.device
        )  # should always be a torch tensor
        next_done = np.zeros(self.n_envs, dtype=bool)  # initialize done to False
        while timesteps_counter < budget:
            obs = next_obs
            done = next_done

            # select action and take step
            with torch.no_grad():
                action, logprobs = self._select_action(obs)
            next_obs, reward, next_terminated, next_truncated, info = self.env.step(
                action
            )
            next_done = np.logical_or(next_terminated, next_truncated)
            next_obs = torch.Tensor(next_obs).to(self.device)

            # end of episode logging
            for i in range(self.n_envs):
                if next_done[i]:
                    self.total_episodes += 1
                    if self.writer and "episode" in info["final_info"][i]:
                        if "episode" in info["final_info"][i]:
                            r, l = (
                                info["final_info"][i]["episode"]["r"],
                                info["final_info"][i]["episode"]["l"],
                            )
                        else:
                            r, l = episode_returns[i], episode_lengths[i]
                        self.writer.add_scalar(
                            "episode_returns", r, self.total_timesteps
                        )
                        self.writer.add_scalar(
                            "episode_lengths", l, self.total_timesteps
                        )
                        self.writer.add_scalar(
                            "total_episodes", self.total_episodes, self.total_timesteps
                        )
                    episode_returns[i], episode_lengths[i] = 0.0, 0

            # append data to memory and update variables
            self.memory.append(
                {
                    "observations": obs.cpu().numpy(),
                    "actions": action,
                    "rewards": reward,
                    "dones": done,
                    "infos": info,
                    "logprobs": logprobs,
                }
            )
            self.total_timesteps += self.n_envs
            timesteps_counter += self.n_envs
            episode_returns += reward
            episode_lengths += 1

            # evaluation
            if (
                self.writer
                and self.eval_freq is not None
                and self.total_timesteps % self.eval_freq == 0
            ):
                evaluation = self.eval(
                    eval_horizon=self.eval_horizon,
                    n_simulations=self.n_eval_episodes,
                    gamma=1.0,
                )
                self.writer.add_scalar("evaluation", evaluation, self.total_timesteps)

            # update with collected experience
            if timesteps_counter % (self.n_envs * self.n_steps) == 0:
                if self.lr_schedule != "constant":
                    lr = lr_scheduler(self.total_timesteps)
                    self.optimizer.param_groups[0]["lr"] = lr
                self._update(next_obs=next_obs, next_done=next_done)

    def _get_lr_scheduler(self, budget):
        """
        Returns a learning rate schedule for the policy and value networks.
        """
        if self.lr_schedule == "constant":
            return lambda t: self.learning_rate
        elif self.lr_schedule == "linear":
            return lambda t: self.learning_rate * (1 - t / float(budget))

    def _select_action(self, obs):
        """
        Select an action given the current state using the policy network.
        Also returns the log probability of the selected action.

        Parameters
        ----------
        obs: torch.Tensor
            Observation tensor of shape (batch_size, obs_dim)

        Returns
        -------
        A tuple (action, log_prob).
        """
        action_dist = self.policy_net(obs)
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)
        return action.cpu().numpy(), action_logprob.cpu().numpy()

    def _update(self, next_obs=None, next_done=None):
        """
        Performs a PPO update based on the data in `self.memory`.

        Parameters
        ----------
        next_obs: torch.Tensor or None
            Next observation tensor of shape (n_envs, obs_dim). Used to
            bootstrap the value function. If None, the value function is
            bootstrapped with zeros.
        next_done: np.ndarray or None
            Array of shape (n_envs,) indicating whether the next observation
            is terminal. If None, this function assumes that they are not
            terminal.

        Notes
        -----
        This function assumes that the data in `self.memory` is complete,
        and it will clear the memory during the update.
        """
        assert (
            int(next_obs is None) + int(next_done is None)
        ) % 2 == 0, "'next_obs' and 'next_done' should be both None or not None at the same time."

        # get batch data
        batch = self.memory.get()
        self.memory.clear()

        # get shapes
        n_steps, n_envs, *obs_shape = batch["observations"].shape
        _, _, *action_shape = batch["actions"].shape

        # create tensors from batch data
        def _to_tensor(x):
            return torch.from_numpy(x).to(self.device).detach()

        b_obs = _to_tensor(batch["observations"])

        # create buffers
        b_values = torch.zeros(
            (n_steps, n_envs), dtype=torch.float32, device=self.device
        )
        b_advantages = torch.zeros_like(b_values)
        b_returns = torch.zeros_like(b_values)

        # compute values
        # note: some implementations compute the value when collecting the data
        #       and use those stale values for the update. This can be better
        #       in architectures with a shared encoder, because you avoid two
        #       forward passes through the encoder. However, we choose to compute
        #       the values here, because it is easier to implement and it has no
        #       impact on performance in most cases.
        with torch.no_grad():
            b_values = self.value_net(b_obs).squeeze(-1)
            if next_obs is not None:
                b_next_value = self.value_net(next_obs).squeeze(-1)

        # compute returns and advantages
        # using numpy and numba for speedup
        rewards = np.copy(batch["rewards"])

        next_dones = np.zeros_like(batch["dones"])
        next_dones[:-1] = batch["dones"][1:]
        if next_obs is not None:
            next_dones[-1] = next_done

        values = b_values.cpu().numpy()
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]
        if next_obs is not None:
            next_values[-1] = b_next_value.cpu().numpy()

        returns = lambda_returns(
            rewards, next_dones, next_values, self.gamma, self.gae_lambda
        )
        advantages = returns - values

        # convert to tensor
        b_actions = _to_tensor(batch["actions"])
        b_logprobs = _to_tensor(batch["logprobs"])
        b_returns = _to_tensor(returns)
        b_advantages = _to_tensor(advantages)

        # flatten the batch
        b_obs = b_obs.view(n_steps * n_envs, *obs_shape)
        b_actions = b_actions.view(n_steps * n_envs, *action_shape)
        b_logprobs = b_logprobs.view(n_steps * n_envs, *action_shape)
        b_values = b_values.view(n_steps * n_envs)
        b_returns = b_returns.view(n_steps * n_envs)
        b_advantages = b_advantages.view(n_steps * n_envs)

        # run minibatch updates
        clipped = []  # whether the policy loss was clipped
        b_indices = np.arange(n_steps * n_envs)
        for epoch in range(self.k_epochs):
            self.rng.shuffle(b_indices)
            for start in range(0, n_steps * n_envs, self.batch_size):
                end = min(start + self.batch_size, n_steps * n_envs)
                mb_indices = b_indices[start:end]

                mb_obs = b_obs[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_old_logprobs = b_logprobs[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_advantages = b_advantages[mb_indices]

                # normalize advantages
                if self.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # forward pass to values and logprobs
                action_dist = self.policy_net(mb_obs)
                mb_values = self.value_net(mb_obs).squeeze(-1)

                mb_logprobs = action_dist.log_prob(mb_actions)
                mb_entropy = action_dist.entropy()
                if len(mb_logprobs.shape) > 1:
                    # in continuous action spaces, the distribution returns one
                    # value per action dim, so we sum over them
                    mb_logprobs = torch.sum(mb_logprobs, dim=-1)
                    mb_old_logprobs = torch.sum(mb_old_logprobs, dim=-1)
                    mb_entropy = torch.sum(mb_entropy, dim=-1)
                mb_logratio = mb_logprobs - mb_old_logprobs
                mb_ratio = torch.exp(mb_logratio)

                # compute approximated kl divergence and whether the policy loss
                # was clipped
                with torch.no_grad():
                    approx_kl = torch.mean((mb_ratio - 1) - mb_logratio)
                    clipped.append(
                        (torch.abs(mb_ratio - 1.0) > self.clip_eps)
                        .float()
                        .mean()
                        .item()
                    )

                # policy loss
                pg_loss1 = -mb_advantages * mb_ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    mb_ratio, 1 - self.clip_eps, 1 + self.clip_eps
                )
                pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

                # value loss
                if self.value_loss == "mse":
                    v_loss = 0.5 * torch.mean((mb_values - mb_returns) ** 2)
                elif self.value_loss == "avec":
                    v_loss = torch.var(mb_returns - mb_values)
                elif self.value_loss == "clipped":
                    mb_old_values = b_values[
                        mb_indices
                    ]  # these are stale after the first minibatch
                    mb_clipped_values = mb_old_values + torch.clamp(
                        mb_values - mb_old_values, -self.clip_eps, self.clip_eps
                    )

                    v_loss_unclipped = (mb_values - mb_returns) ** 2
                    v_loss_clipped = (mb_clipped_values - mb_returns) ** 2
                    v_loss = 0.5 * torch.mean(
                        torch.max(v_loss_unclipped, v_loss_clipped)
                    )

                # entropy loss
                entropy_loss = torch.mean(mb_entropy)

                # total loss
                loss = pg_loss + self.vf_coef * v_loss - self.entr_coef * entropy_loss

                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        list(self.policy_net.parameters())
                        + list(self.value_net.parameters()),
                        self.max_grad_norm,
                    )
                self.optimizer.step()

            if self.target_kl and approx_kl > self.target_kl:
                break

        # compute explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # log metrics
        # note: this approach only logs the last batch of the last
        # epoch, which is not ideal. However, it is the way it is
        # done in most implementations of PPO.
        if self.writer:
            self.writer.add_scalar(
                "fit/policy_loss",
                pg_loss.item(),
                self.total_timesteps,
            )
            self.writer.add_scalar(
                "fit/value_loss",
                v_loss.item(),
                self.total_timesteps,
            )
            self.writer.add_scalar(
                "fit/entropy_loss",
                entropy_loss.item(),
                self.total_episodes,
            )
            self.writer.add_scalar(
                "fit/approx_kl",
                approx_kl.item(),
                self.total_episodes,
            )
            self.writer.add_scalar(
                "fit/clipfrac",
                np.mean(clipped),
                self.total_episodes,
            )
            self.writer.add_scalar(
                "fit/explained_variance",
                explained_var,
                self.total_episodes,
            )
            self.writer.add_scalar(
                "fit/learning_rate",
                self.optimizer.param_groups[0]["lr"],
            )

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        batch_size = trial.suggest_categorical("batch_size", [1, 4, 8, 16, 32])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        entr_coef = trial.suggest_float("entr_coef", 1e-8, 0.1, log=True)

        clip_eps = trial.suggest_categorical("clip_eps", [0.1, 0.2, 0.3])

        k_epochs = trial.suggest_categorical("k_epochs", [1, 5, 10, 20])

        return {
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "entr_coef": entr_coef,
            "clip_eps": clip_eps,
            "k_epochs": k_epochs,
        }

    ##### Overwrite some inherited functions

    def save(self, filename):
        # Overwrite the 'save' and 'load' functions to not store the env if it's a "vectorized env" (can't be managed with pickle)
        # If overridden, the load() method must also be overriden.
        """
        Save agent object. By default, the agent is pickled.

        If overridden, the load() method must also be overriden.

        Before saving, consider setting writer to None if it can't be pickled (tensorboard writers
        keep references to files and cannot be pickled).

        Note: dill[1]_ is used when pickle fails
        (see https://stackoverflow.com/a/25353243, for instance).
        Pickle is tried first, since it is faster.

        Parameters
        ----------
        filename: Path or str
            File in which to save the Agent.

        Returns
        -------
        pathlib.Path
            If save() is successful, a Path object corresponding to the filename is returned.
            Otherwise, None is returned.
        .. warning:: The returned filename might differ from the input filename: For instance,
        the method can append the correct suffix to the name before saving.

        References
        ----------
        .. [1] https://github.com/uqfoundation/dill
        """
        # remove writer if not pickleable
        if not dill.pickles(self.writer):
            self.set_writer(None)
        # save
        filename = Path(filename).with_suffix(".pickle")
        filename.parent.mkdir(parents=True, exist_ok=True)

        dict_to_save = dict(self.__dict__)
        del dict_to_save["env"]
        del dict_to_save["eval_env"]

        try:
            if not self.compress_pickle:
                with filename.open("wb") as ff:
                    pickle.dump(dict_to_save, ff)
            else:
                with bz2.BZ2File(filename, "wb") as ff:
                    cPickle.dump(dict_to_save, ff)
        except Exception:
            try:
                if not self.compress_pickle:
                    with filename.open("wb") as ff:
                        dill.dump(dict_to_save, ff)
                else:
                    with bz2.BZ2File(filename, "wb") as ff:
                        dill.dump(dict_to_save, ff)
            except Exception as ex:
                logger.warning("Agent instance cannot be pickled: " + str(ex))
                return None

        return filename

    @classmethod
    def load(cls, filename, **kwargs):
        """
        Overwrite the 'save' and 'load' functions to not store the env if it's a "vectorized env" (can't be managed with pickle)

        ----- documentation from original load -----
        Load agent object.
        If overridden, save() method must also be overriden.

        Parameters
        ----------
        **kwargs: dict
            Arguments to required by the __init__ method of the Agent subclass.
        """
        filename = Path(filename).with_suffix(".pickle")
        obj = cls(**kwargs)

        try:
            if not obj.compress_pickle:
                with filename.open("rb") as ff:
                    tmp_dict = pickle.load(ff)
            else:
                with bz2.BZ2File(filename, "rb") as ff:
                    tmp_dict = cPickle.load(ff)
        except Exception:
            if not obj.compress_pickle:
                with filename.open("rb") as ff:
                    tmp_dict = dill.load(ff)
            else:
                with bz2.BZ2File(filename, "rb") as ff:
                    tmp_dict = dill.load(ff)

        temp_env = obj.__dict__["env"]
        temp_eval_env = obj.__dict__["eval_env"]

        obj.__dict__.clear()
        obj.__dict__.update(tmp_dict)

        obj.__dict__["env"] = temp_env
        obj.__dict__["eval_env"] = temp_eval_env

        return obj
