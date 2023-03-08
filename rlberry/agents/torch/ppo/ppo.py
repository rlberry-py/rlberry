import numpy as np
import torch
import torch.nn as nn


import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy

# from rlberry.agents.utils.memories import Memory

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

import rlberry

logger = rlberry.logger


# Notes about VecEnvs:
# - reset() returns a numpy array of shape (n_envs, state_dim)
# - step() returns a tuple of arrays (states, rewards, dones, infos)
#   - states: np.array (n_envs, state_dim) dtype varies
#   - rewards: np.array (n_envs,) np.float64
#   - dones: np.array (n_envs,) bool
#   - infos: list (n_envs,) dict
# - close() closes all environments


class PPOAgent(AgentWithSimplePolicy):
    """
    Proximal Policy Optimization Agent.

    Policy gradient methods for reinforcement learning, which alternate between
    sampling data through interaction with the environment, and optimizing a
    “surrogate” objective function using stochastic gradient ascent

    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
    batch_size : int
        Size of mini batches during the k_epochs gradient descent steps.
    n_steps : int
        Number of transitions to use for parameters updates.
    gamma : double
        Discount factor in [0, 1].
    entr_coef : double
        Entropy coefficient.
    vf_coef : double
        Value function loss coefficient.
    learning_rate : double
        Learning rate.
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    clip_eps : double
        PPO clipping range (epsilon).
    k_epochs : int
        Number of epochs per update.
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
    normalize_rewards : bool
        whether or not to normalize rewards
    normalize_avantages : bool
        whether or not to normalize advantages
    device: str
        Device to put the tensors on


    References
    ----------
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017).
    "Proximal Policy Optimization Algorithms."
    arXiv preprint arXiv:1707.06347.

    Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015).
    "Trust region policy optimization."
    In International Conference on Machine Learning (pp. 1889-1897).
    """

    name = "PPO"
    __value_losses__ = ["clipped", "mse", "avec"]
    __lr_schedule___ = ["constant", "linear"]

    def __init__(
        self,
        env,
        n_envs=1,
        batch_size=32,
        n_steps=512,
        gamma=0.99,
        entr_coef=0.01,
        vf_coef=0.5,
        learning_rate=3e-4,
        lr_schedule="constant",
        optimizer_type="ADAM",
        value_loss="mse",
        clip_eps=0.2,
        k_epochs=10,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        target_kl=0.05,
        normalize_advantages=True,
        policy_net_fn=None,
        value_net_fn=None,
        policy_net_kwargs=None,
        value_net_kwargs=None,
        n_eval_episodes=10,
        eval_horizon=int(1e5),
        eval_freq=None,
        device="cuda:best",
        eval_env=None,
        **kwargs
    ):
        kwargs.pop("eval_env", None)
        AgentWithSimplePolicy.__init__(
            self, None, **kwargs
        )  # PPO handles the env internally

        # create environment
        self.n_envs = n_envs
        self.env = process_ppo_env(env, self.seeder, n_envs)
        eval_env = eval_env or env
        self.eval_env = process_env(eval_env, self.seeder, copy_env=True)

        # hyperparameters
        assert value_loss in self.__value_losses__, "value_loss must be in {}".format(
            self.__value_losses__
        )
        assert lr_schedule in self.__lr_schedule___, "lr_schedule must be in {}".format(
            self.__lr_schedule___
        )

        self.batch_size = batch_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.entr_coef = entr_coef
        self.vf_coef = vf_coef
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.optimizer_type = optimizer_type
        self.value_loss = value_loss
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.n_eval_episodes = n_eval_episodes
        self.eval_horizon = eval_horizon
        self.eval_freq = eval_freq
        self.kwargs = kwargs

        self.normalize_advantages = normalize_advantages
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

        self.optimizer_kwargs = {"optimizer_type": optimizer_type, "lr": learning_rate}

        # loss function
        if self.value_loss == "mse":
            self._loss = nn.MSELoss()
        elif self.value_loss == "avec":
            raise NotImplementedError("Avec loss not implemented yet.")

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
        self.total_timesteps = 0
        self.total_episodes = 0

        # Initialize rollout buffer
        # TODO: change states to observations
        self.memory = RolloutBuffer(self.rng, self.n_steps)
        self.memory.setup_entry("observations", dtype=np.float32)
        self.memory.setup_entry("actions", dtype=self.env.single_action_space.dtype)
        self.memory.setup_entry("rewards", dtype=np.float32)
        self.memory.setup_entry("dones", dtype=bool)
        self.memory.setup_entry("action_logprobs", dtype=np.float32)
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
            are updated every n_steps steps using n_steps//batch_size mini batches.
        """
        del kwargs

        if lr_scheduler is None:
            lr_scheduler = self._get_lr_scheduler(budget)
        eval_freq = self.eval_freq or (budget // 10)
        timesteps_counter = 0

        episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
        episode_lengths = np.zeros(self.n_envs, dtype=np.int32)

        next_obs = torch.Tensor(self.env.reset()).to(
            self.device
        )  # should always be a torch tensor
        next_done = np.zeros(self.n_envs, dtype=bool)  # initialize done to False
        while timesteps_counter < budget:
            obs = next_obs
            done = next_done

            # select action and take step
            with torch.no_grad():
                action, logprobs = self._select_action(obs)
            next_obs, reward, next_done, info = self.env.step(action)
            next_obs = torch.Tensor(next_obs).to(self.device)

            # append data to memory and update variables
            self.memory.append(
                {
                    "observations": obs.cpu().numpy(),
                    "actions": action,
                    "rewards": reward,
                    "dones": done,
                    "infos": info,
                    "action_logprobs": logprobs,
                }
            )
            self.total_timesteps += self.n_envs
            timesteps_counter += self.n_envs
            episode_rewards += reward
            episode_lengths += 1

            # end of episode logging
            for i in range(self.n_envs):
                if done[i]:
                    self.total_episodes += 1
                    if self.writer:
                        self.writer.add_scalar(
                            "episode_rewards", episode_rewards[i], self.total_timesteps
                        )
                        self.writer.add_scalar(
                            "episode_lengths", episode_lengths[i], self.total_timesteps
                        )
                        self.writer.add_scalar(
                            "total_episodes", self.total_episodes, self.total_timesteps
                        )
                    episode_rewards[i], episode_lengths[i] = 0.0, 0

            # evaluation
            if self.writer and self.total_timesteps % eval_freq == 0:
                evaluation = self.eval(
                    eval_horizon=self.eval_horizon, n_simulations=self.n_eval_episodes
                )
                self.writer.add_scalar(
                    "fit/evaluation", evaluation, self.total_timesteps
                )

            # update with collected experience
            if timesteps_counter % (self.n_envs * self.n_steps) == 0:
                if self.lr_schedule != "constant":
                    lr = lr_scheduler(timesteps_counter)
                    self.optimizer.param_groups[0]["lr"] = lr
                self._update(next_obs=next_obs, next_done=next_done)

    def _get_lr_scheduler(self, budget):
        """
        Returns a learning rate schedule for the policy and value networks.
        """
        if self.lr_schedule == "constant":
            return lambda t: self.learning_rate
        elif self.lr_schedule == "linear":
            return lambda t: self.learning_rate * (1 - t / budget)

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
            b_values = self.value_net(b_obs.view(n_steps * n_envs, *obs_shape)).view(
                n_steps, n_envs
            )
            if next_obs is not None:
                b_next_value = torch.squeeze(self.value_net(next_obs))

        # compute returns and advantages
        # using numpy and numba for speedup
        rewards = np.copy(batch["rewards"])

        next_dones = np.zeros_like(batch["dones"])
        next_dones[:-1] = batch["dones"][1:]
        if next_obs is not None:
            next_dones[-1] = next_done

        next_values = np.zeros_like(batch["rewards"])
        next_values[:-1] = b_values[1:].cpu().numpy()
        if next_obs is not None:
            next_values[-1] = b_next_value.cpu().numpy()

        returns = lambda_returns(
            rewards, next_dones, next_values, self.gamma, self.gae_lambda
        )
        advantages = returns - b_values.cpu().numpy()

        # convert to tensor
        b_actions = _to_tensor(batch["actions"])
        b_logprobs = _to_tensor(batch["action_logprobs"])
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
                end = start + self.batch_size
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
                mb_values = self.value_net(mb_obs)

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
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)

        entr_coef = trial.suggest_loguniform("entr_coef", 1e-8, 0.1)

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


# if __name__ == "__main__":
#     env = (gym_make, dict(id="Acrobot-v1"))
#     # env = gym_make(id="Acrobot-v1")
#     ppo = PPOAgent(env)
#     ppo.fit(100000)
